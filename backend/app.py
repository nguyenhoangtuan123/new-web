from flask import Flask, request, jsonify
from firebase_functions import https_fn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import scipy.optimize as opt
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
import xgboost as xgb
from shap_xgboost import get_shap_values_for_instance
from lime_xgboost import get_lime_explanations_for_instance
import joblib
import pandas as pd
from flask_cors import CORS
import os
from cvxopt import matrix, solvers
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)  # Cho phép CORS từ mọi nguồn

# Định nghĩa GradientBoostingClassifier
gbc = GradientBoostingClassifier(
    n_estimators=3,  # Số giai đoạn boosting
    learning_rate=0.05,  # Giá trị nhỏ để tránh overfitting
    max_depth=5,  # Độ sâu cây
    subsample=0.8,  # Tăng khả năng tổng quát hóa
    random_state=42
)

# Lớp MyLogisticRegression (giữ nguyên)
class MyLogisticRegression:
    def __init__(self, C=1.0):
        self.C = C
        self.coef_ = None

    def _softmax(self, X, beta):
        beta = beta.reshape(self.n_classes_, self.n_features_)
        logits = X @ beta.T
        return softmax(logits, axis=1)

    def _log_likelihood(self, beta, X, y):
        beta = beta.reshape(self.n_classes_, self.n_features_)
        logits = X @ beta.T
        log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        reg_term = (self.C / 2) * np.sum(beta[:, 1:] ** 2)  # L2 regularization term
        return -np.sum(y * log_probs) + reg_term

    def _likelihood_gradient(self, beta, X, y):
        beta = beta.reshape(self.n_classes_, self.n_features_)
        probs = self._softmax(X, beta)
        reg_grad = self.C * beta[:, 1:]  # L2 
        reg_grad = np.concatenate([np.zeros((beta.shape[0], 1)), reg_grad], axis=1)
        return ((X.T @ (probs - y)).T + reg_grad).flatten()
        
    def fit(self, X, y):
        y = y.astype(int)
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Thêm intercept
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        
        y_one_hot = np.eye(self.n_classes_)[y]
        self.y_one_hot = y_one_hot
        beta_init = np.zeros((self.n_classes_, self.n_features_)).flatten()
        
        result = opt.minimize(
            self._log_likelihood,
            beta_init,
            args=(X, y_one_hot),
            method='BFGS',
            jac=self._likelihood_gradient,
            options={'disp': False}
        )
        
        self.coef_ = result.x.reshape(self.n_classes_, self.n_features_)
        return self

    def decision_function(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])  # Thêm intercept
        return X @ self.coef_.T

    def predict_proba(self, X):
        return self._softmax(X, self.coef_)

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.argmax(self.predict_proba(X), axis=1)

    def get_coef(self):
        return self.coef_

# Hàm sigmoid (giữ nguyên)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Lớp SVMQP (giữ nguyên)
class SVMQP(BaseEstimator, ClassifierMixin):
    def __init__(self, epsilon=1e-5, C=100, kernel='linear', gamma=0.02, class_weight=None):
        self.lambdas = None
        self.epsilon = epsilon
        self.C = C
        assert kernel in ['linear', 'rbf'], "Vui lòng chọn kernel hợp lệ"
        self.kernel = kernel
        self.gamma = gamma  # cho kernel = 'rbf'
        self.class_weight = class_weight  # mặc định class_weight=None
    
    def fit(self, X, y):
        if self.gamma == 'scale':  # giống sklearn
            n_features = X.shape[1]
            var_X = X.var()
            self._gamma = 1.0 / (n_features * var_X) if var_X > 0 else 1.0
        else:
            self._gamma = self.gamma
            
        self.X = np.array(X)
        self.y = np.array(2 * y - 1).astype(np.double)
        N = self.X.shape[0] 
        V = self.X * np.expand_dims(self.y, axis=1)

        # Tính trọng số lớp cho chế độ 'balanced'
        unique_classes, class_counts = np.unique(y, return_counts=True)
        num_classes = len(unique_classes)
        if self.class_weight == 'balanced':
            class_weight_dict = {cls: N / (num_classes * count) for cls, count in zip(unique_classes, class_counts)}
        elif isinstance(self.class_weight, np.ndarray):
            assert len(self.class_weight) == num_classes, "Kích thước class_weight phải khớp với số lớp"
            class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, self.class_weight)}
        else:
            class_weight_dict = {cls: 1.0 for cls in unique_classes}

        sample_weights = np.array([class_weight_dict[yi] for yi in y])

        if self.kernel == 'rbf':
            K = matrix(np.outer(self.y, self.y) * self.rbf_kernel(self.X, self.X))
        else:
            K = matrix(V.dot(V.T))
        
        p = matrix(-np.ones((N, 1)))
        G = matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = matrix(np.vstack((np.zeros((N, 1)), (self.C * sample_weights).reshape(N, 1))))
        A = matrix(self.y.reshape(-1, N))
        b = matrix(np.zeros((1, 1)))
        
        solvers.options['show_progress'] = False
        print('Đang giải QP')
        sol = solvers.qp(K, p, G, h, A, b)
        self.lambdas = np.array(sol['x'])
        self.get_wb()
        
    def rbf_kernel(self, X1, X2):
        sq_dists = cdist(X1, X2, 'sqeuclidean')
        return np.exp(-1.0 * self._gamma * sq_dists)
        
    def get_lambdas(self):
        return self.lambdas

    def get_wb(self):
        S = np.where(self.lambdas > self.epsilon)[0]
        V = self.X * np.expand_dims(self.y, axis=1)

        VS = V[S, :]
        XS = self.X[S, :]
        yS = self.y[S]
        lS = self.lambdas[S]

        self.XS = XS
        self.yS = yS
        self.lS = lS
        
        if self.kernel == 'rbf':
            alpha = lS * np.expand_dims(yS, axis=1)
            b = np.mean(np.expand_dims(yS, axis=1) - self.rbf_kernel(XS, XS).dot(alpha))
            self.b = b
            return b
        else:
            w = lS.T.dot(VS)
            b = np.mean(np.expand_dims(yS, axis=1) - XS.dot(w.T))
            self.w = w
            self.b = b
            return self.w, self.b
    
    def print_lambdas(self):
        print('lambda = ')
        print(self.lambdas.T)
        S = np.where(self.lambdas > self.epsilon)[0]
        print(self.lambdas[S])

    def predict(self, X_test):
        K_test = self.rbf_kernel(X_test, self.XS)
        conf = K_test @ (self.lS * np.expand_dims(self.yS, axis=1)) + self.b
        return (np.squeeze(np.sign(conf)) + 1) // 2

    def decision_function(self, X_test):
        K_test = self.rbf_kernel(X_test, self.XS)
        conf = K_test @ (self.lS * np.expand_dims(self.yS, axis=1)) + self.b
        return np.squeeze(conf)
    
    def predict_proba(self, X_test):
        assert False, "Lớp này không hỗ trợ xác suất"
        K_test = self.rbf_kernel(X_test, self.XS)
        conf = K_test @ (self.lS * np.expand_dims(self.yS, axis=1)) + self.b

# In thư mục hiện tại
print("Thư mục hiện tại:", os.getcwd())

# Định nghĩa tên các mô hình
model_names = [
    'LogisticRegression_scratch',
    'SVC_scratch',
    'GradientBoostingClassifier_SMOTE',
    'XGBoost_SMOTE',
    'SVC_SMOTE',
    'LogisticRegression_SMOTE',
    'GradientBoostingClassifier_FFT',
    'XGBoost_FFT',
    'LogisticRegression_FFT',
    'SVC_FFT',
    'GradientBoostingClassifier_classweight',
    'XGBoost_classweight',
    'LogisticRegression_classweight',
    'SVC_classweight',
]

# Load scaler
def process_mitbih_fft():
    try:
        print("Đang load mitbih_train.csv...")
        df = pd.read_csv("mitbih_train.csv", header=None)
        print(f"Đã load dataset với shape: {df.shape}")
        
        X = df.iloc[:, :-1].values
        print("Đang áp dụng biến đổi FFT...")
        X_fft = np.fft.fft(X, axis=1)
        X_fft_magnitude = np.abs(X_fft[:, :X_fft.shape[1] // 2])
        
        print("Đang fit standard scaler...")
        fft_scaler = StandardScaler()
        fft_scaler.fit(X_fft_magnitude)
        print("Shape của dữ liệu FFT:", X_fft_magnitude.shape)
        
        scaler_filename = "fft_scaler.pkl"
        joblib.dump(fft_scaler, scaler_filename)
        print(f"Đã lưu StandardScaler vào {scaler_filename}")
        return fft_scaler
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy mitbih_train.csv")
        return None
    except Exception as e:
        print(f"Lỗi khi xử lý dữ liệu mitbih: {str(e)}")
        return None

try:
    fft_scaler = joblib.load("fft_scaler.pkl")
    print("Đã load fft_scaler.pkl thành công")
except FileNotFoundError:
    print("Không tìm thấy fft_scaler.pkl, đang tạo mới...")
    fft_scaler = process_mitbih_fft()

def process_mitbih():
    try:
        print("Đang load mitbih_train.csv...")
        df = pd.read_csv("mitbih_train.csv", header=None)
        print(f"Đã load dataset với shape: {df.shape}")
        
        X = df.iloc[:, :-1].values
        scaler = StandardScaler()
        scaler.fit(X)
        
        scaler_filename = "scaler.pkl"
        joblib.dump(scaler, scaler_filename)
        print(f"Đã lưu StandardScaler vào {scaler_filename}")
        return scaler
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy mitbih_train.csv")
        return None
    except Exception as e:
        print(f"Lỗi khi xử lý dữ liệu mitbih: {str(e)}")
        return None

try:
    scaler = joblib.load("scaler.pkl")
    print("Đã load scaler.pkl thành công")
except FileNotFoundError:
    print("Không tìm thấy scaler.pkl, đang tạo mới...")
    scaler = process_mitbih()

# Load các mô hình
models = []
for model_name in model_names:
    try:
        model = joblib.load(f"{model_name}.pkl")
        print(f"Đã load {model_name}.pkl thành công")
        models.append(model)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy {model_name}.pkl")
        models.append(None)
    except Exception as e:
        print(f"Lỗi khi load {model_name}.pkl: {str(e)}")
        models.append(None)

# Load dữ liệu mẫu
try:
    df = pd.read_csv("mitbih_train.csv", header=None, on_bad_lines='skip')
    print(f"Đã load {len(df)} hàng từ mitbih_train.csv")
    df = df.dropna()
    print(f"Sau khi dropna: {len(df)} hàng")

    if not df.empty:
        label_col = df.columns[-1]
        class_counts = df[label_col].value_counts()
        if len(class_counts) < 5:
            raise ValueError("Tìm thấy ít hơn 5 lớp trong dữ liệu")
        
        samples_per_class = 30 // 5
        sampled_df = df.groupby(label_col).apply(
            lambda g: g.sample(n=min(samples_per_class, len(g)), random_state=42)
        ).reset_index(drop=True)

        if len(sampled_df) < 30:
            needed = 30 - len(sampled_df)
            extras = df[df[label_col].isin(class_counts[class_counts > samples_per_class].index)]
            extra_samples = extras.sample(n=needed, random_state=42)
            sampled_df = pd.concat([sampled_df, extra_samples], ignore_index=True)

        sample_data = sampled_df.values
        print(f"Đã lấy mẫu {len(sample_data)} hàng với 5 lớp")
    else:
        sample_data = np.array([])
        print("Lỗi: mitbih_train.csv không có dữ liệu hợp lệ")
except FileNotFoundError:
    sample_data = np.array([])
    print("Lỗi: Không tìm thấy mitbih_train.csv")
except Exception as e:
    sample_data = np.array([])
    print(f"Lỗi khi load dữ liệu: {e}")

# Endpoint /get_samples
@app.route("/get_samples", methods=["GET"])
def get_samples():
    if not sample_data.size:
        return jsonify({"error": "Không có dữ liệu mẫu"}), 404
    
    samples = []
    for i, row in enumerate(sample_data):
        if len(row) < 188:
            continue
        signal = row[:-1]
        if not all(isinstance(x, (int, float)) for x in signal):
            print(f"Dữ liệu tín hiệu không hợp lệ ở hàng {i}")
            continue
        try:
            label = int(row[-1])
        except ValueError:
            print(f"Nhãn không hợp lệ ở hàng {i}: {row[-1]}")
            continue
        labels = ["Normal", "Supraventricular Ectopic", "Ventricular Ectopic", "Fusion", "Unknown"]
        samples.append({
            "id": i,
            "label": labels[label],
            "signal": signal.tolist()
        })
    if not samples:
        return jsonify({"error": "Không có dữ liệu mẫu hợp lệ"}), 404
    return jsonify({"samples": samples})

# Endpoint /classify
@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.json.get('signal')
        if not data or not isinstance(data, list):
            return jsonify({"error": "Dữ liệu tín hiệu không hợp lệ"}), 400
        
        if len(data) != 187:
            return jsonify({"error": "Tín hiệu phải có độ dài 187"}), 400
        
        if not all(isinstance(x, (int, float)) for x in data):
            return jsonify({"error": "Tín hiệu chỉ được chứa số"}), 400
        
        if scaler is None or fft_scaler is None:
            return jsonify({"error": "Scaler không được load"}), 500

        data = np.array(data).reshape(1, -1)
        labels = ["Normal", "Supraventricular Ectopic", "Ventricular Ectopic", "Fusion", "Unknown"]
        results = []
        
        try:
            data_scaled = scaler.transform(data)
            X_fft = np.fft.fft(data, axis=1)
            X_fft_magnitude = np.abs(X_fft[:, :X_fft.shape[1] // 2])
            fft_data = fft_scaler.transform(X_fft_magnitude)
            print(data_scaled)
        except Exception as e:
            print(f"Lỗi scaler với shape dữ liệu {data.shape}: {str(e)}")
            return jsonify({"error": f"Lỗi khi transform scaler: {str(e)}"}), 500

        for i, (model, model_name) in enumerate(zip(models, model_names)):
            if model is None:
                results.append({
                    "model": model_name,
                    "error": f"Mô hình {model_name} không được load"
                })
                continue
            
            try:
                if model_name in ["XGBoost_SMOTE", "XGBoost_classweight"]:
                    Ddata_scaled = xgb.DMatrix(data_scaled)
                    prediction = model.predict(Ddata_scaled)[0]
                elif model_name == "XGBoost_FFT":
                    Dfft_data = xgb.DMatrix(fft_data)
                    prediction = model.predict(Dfft_data)[0]
                elif model_name.endswith("FFT"):
                    prediction = model.predict(fft_data)[0]
                else:
                    prediction = model.predict(data_scaled)[0]
                try:
                    pred_int = int(prediction)
                    if pred_int < 0 or pred_int >= len(labels):
                        raise ValueError(f"Dự đoán {pred_int} ngoài phạm vi nhãn")
                    result = labels[pred_int]
                except (ValueError, TypeError) as e:
                    results.append({
                        "model": model_name,
                        "error": f"Dự đoán không hợp lệ: {str(e)}"
                    })
                    continue
                
                try:
                    print(model_name)
                    if model_name == "LogisticRegression_scratch":
                        data_scaled_tmp = np.hstack([np.ones((data_scaled.shape[0], 1)), data_scaled])
                        probabilities = model.predict_proba(data_scaled_tmp)[0].tolist()
                    elif model_name in ["XGBoost_SMOTE", "XGBoost_classweight"]:
                        probabilities = model.predict_proba(xgb.DMatrix(data_scaled))[0].tolist()
                    elif model_name in ["SVC_scratch", "SVC_SMOTE", "SVC_classweight"]:
                        probabilities = model.decision_function(data_scaled)[0].tolist()
                        probabilities = softmax(probabilities).tolist()
                    elif model_name == "XGBoost_FFT":
                        probabilities = model.predict_proba(xgb.DMatrix(fft_data))[0].tolist()
                    elif model_name == "SVC_FFT":
                        probabilities = model.decision_function(fft_data)[0].tolist()
                        probabilities = softmax(probabilities).tolist()
                    elif model_name.endswith("FFT"):
                        probabilities = model.predict_proba(fft_data)[0].tolist()
                    else:
                        probabilities = model.predict_proba(data_scaled)[0].tolist()
                    results.append({
                        "model": model_name,
                        "prediction": result,
                        "probabilities": probabilities
                    })
                except AttributeError:
                    print(f"Mô hình {model_name} không hỗ trợ predict_proba")
                    results.append({
                        "model": model_name,
                        "prediction": result,
                        "error": "Mô hình không hỗ trợ predict_proba"
                    })
            except Exception as e:
                print(f"Lỗi với mô hình {model_name}: {str(e)}")
                results.append({
                    "model": model_name,
                    "error": str(e)
                })
        
        return jsonify({"results": results})
    except Exception as e:
        print(f"Lỗi ở endpoint classify: {str(e)}")
        return jsonify({"error": str(e)}), 400

# Endpoint /get_shape_xgboost
@app.route('/get_shape_xgboost', methods=['POST'])
def get_shape_xgboost():
    try:
        data = request.json.get("signal")
        if not data:
            return jsonify({"error": "Không có dữ liệu cung cấp"}), 400

        shap_values, scaled_instance, class_idx = get_shap_values_for_instance(data)
        shap_values = np.transpose(shap_values)  # shape = (5, 187)
        return jsonify({
            "shap_values": shap_values.tolist(),
            "scaled_instance": scaled_instance.tolist(),
            "class_idx": int(class_idx)
        })
    except Exception as e:
        print(f"Lỗi ở endpoint get_shape_xgboost: {str(e)}")
        return jsonify({"error": str(e)}), 400

# Endpoint /get_lime_xgboost
@app.route('/get_lime_xgboost', methods=['POST'])
def get_shape_xgboost_lime():
    try:
        data = request.json.get("signal")
        if not data:
            return jsonify({"error": "Không có dữ liệu cung cấp"}), 400

        scaled_instance, lime_values, class_idx = get_lime_explanations_for_instance(data)
        scaled_instance = np.array(scaled_instance) / 500
        return jsonify({
            "Lime_values": lime_values.tolist(),
            "scaled_instance": scaled_instance.tolist(),
            "class_idx": int(class_idx)
        })
    except Exception as e:
        print(f"Lỗi ở endpoint get_lime_xgboost: {str(e)}")
        return jsonify({"error": str(e)}), 400

# Hàm chính cho Cloud Functions
@https_fn.on_request()
def main(req: https_fn.Request) -> https_fn.Response:
    from wsgiref.handlers import CGIHandler
    return CGIHandler().run(app)