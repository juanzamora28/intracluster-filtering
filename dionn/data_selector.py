import numpy as np
from collections import Counter
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
import tensorflow as tf
from scipy import linalg, special
import warnings
from scipy.optimize import linear_sum_assignment
import torch

class StudentMixture:
    """Modelo de mezcla de distribuciones t de Student."""
    def __init__(self, n_components, covariance_type='full', tol=1e-3, reg_covar=1e-6, max_iter=100, random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_state)
        self.converged_ = False

    def _initialize_parameters(self, X):
        """Inicializa los parámetros del modelo de mezcla."""
        n_samples, n_features = X.shape
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        self.means_ = X[self.random_state.choice(n_samples, self.n_components, replace=False)]
        self.covariances_ = np.array([np.cov(X.T) + self.reg_covar * np.eye(n_features) for _ in range(self.n_components)])
        self.degrees_of_freedom_ = np.full(self.n_components, 10.0)

    def _estimate_log_prob(self, X):
        """Estima las probabilidades logarítmicas para cada componente."""
        n_samples, n_features = X.shape
        log_prob = np.empty((n_samples, self.n_components))

        for k in range(self.n_components):
            nu = self.degrees_of_freedom_[k]
            if nu <= 0:
                raise ValueError(f"Grados de libertad no válidos: {nu}. Deben ser mayores que 0.")
            diff = X - self.means_[k]
            precision = linalg.inv(self.covariances_[k])
            quad_form = np.sum(diff @ precision * diff, axis=1)
            log_det_cov = np.log(max(linalg.det(self.covariances_[k]), 1e-10))  # Evitar log(0) o negativos

            log_prob[:, k] = (
                special.gammaln((nu + n_features) / 2)
                - special.gammaln(nu / 2)
                - 0.5 * (n_features * np.log(nu * np.pi) + log_det_cov)
                - 0.5 * (nu + n_features) * np.log(1 + np.maximum(quad_form / nu, 1e-10))
            )
        return log_prob

    def _e_step(self, X):
        """Paso E."""
        log_prob = self._estimate_log_prob(X)
        weighted_log_prob = log_prob + np.log(self.weights_)
        log_prob_norm = special.logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return np.mean(log_prob_norm), log_resp

    def _m_step(self, X, log_resp):
        """Paso M."""
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.weights_ = nk / nk.sum()
        self.means_ = np.dot(resp.T, X) / nk[:, np.newaxis]
        self.covariances_ = self._estimate_covariances(X, resp, nk)
        self.degrees_of_freedom_ = self._update_degrees_of_freedom(X, resp, nk)

    def _estimate_covariances(self, X, resp, nk):
        """Estima las matrices de covarianza para cada componente."""
        n_samples, n_features = X.shape
        covariances = np.empty((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_diff = resp[:, k][:, np.newaxis] * diff
            covariances[k] = np.dot(weighted_diff.T, diff) / nk[k] + self.reg_covar * np.eye(n_features)
        return covariances

    def _update_degrees_of_freedom(self, X, resp, nk):
        """Actualiza los grados de libertad para cada componente."""
        n_samples, n_features = X.shape
        new_dof = np.empty(self.n_components)
        for k in range(self.n_components):
            diff = X - self.means_[k]
            quad_form = np.sum(diff @ linalg.inv(self.covariances_[k]) * diff, axis=1)
            weighted_quad_form = np.dot(resp[:, k], quad_form)
            new_dof[k] = max(2 * (n_features + nk[k]) / (nk[k] - weighted_quad_form / (self.degrees_of_freedom_[k] + 2)), 1.0)
        return new_dof

    def fit(self, X):
        """Estimación de parámetros con el algoritmo EM."""
        self._initialize_parameters(X)
        for n_iter in range(self.max_iter):
            prev_weights = self.weights_.copy()
            log_prob_norm, log_resp = self._e_step(X)
            self._m_step(X, log_resp)
            if np.allclose(self.weights_, prev_weights, atol=self.tol):
                self.converged_ = True
                print(f"Convergencia alcanzada en la iteración {n_iter}.")
                break

    def predict_proba(self, X):
        """Calcula las probabilidades posteriores para cada componente."""
        _, log_resp = self._e_step(X)
        return np.exp(log_resp)

class DataSelector:
    def __init__(self, X_tr, y_tr, epochs_to_start_filter, update_period_in_epochs, filter_percentile=0.3,
                 random_state=None, train_with_outliers=False, filter_model="gmm"):
        self.X_tr = X_tr
        self.y_tr = y_tr.numpy() if isinstance(y_tr, tf.Tensor) else y_tr
        self.out_clases_number = self.y_tr.shape[1]
        self.epochs_to_start_filter = epochs_to_start_filter
        self.update_period_in_epochs = update_period_in_epochs
        self.filtered_index = list(range(X_tr.shape[0]))
        self.filter_percentile = filter_percentile
        self.random_state = random_state
        self.train_with_outliers = train_with_outliers
        self.removed_data_indices = []
        self.original_indices = np.arange(X_tr.shape[0])  # Índices originales
        self.all_removed_indices = []  # Lista de todos los índices removidos
        self.previous_X_tr = X_tr  # Copia de datos inicial
        self.previous_y_tr = y_tr
        self.inspector_layer_out = []  # Para almacenar la salida del inspector
        self.filter_model = filter_model.lower()  # 'gmm' o 'smm'

    def check_filter_update_criteria(self, epoch):
        return (epoch >= self.epochs_to_start_filter and
                (epoch - self.epochs_to_start_filter) % self.update_period_in_epochs == 0)

    def apply_pca(self, inspector_layer_out, explained_variance=None, n_components=None):
        if explained_variance is not None:
            pca = PCA(n_components=explained_variance)
        elif n_components is not None:
            pca = PCA(n_components=n_components)
        else:
            raise ValueError("Debes proporcionar explained_variance o n_components.")
        transformed_out = pca.fit_transform(inspector_layer_out)
        n_components = transformed_out.shape[1]
        if n_components < 2:
            n_components = 2
            pca = PCA(n_components=n_components)
            transformed_out = pca.fit_transform(inspector_layer_out)
        if explained_variance is not None:
            print(f"PCA realizado: se retuvo el {explained_variance*100}% de la varianza con {n_components} componentes.")
        else:
            print(f"PCA realizado con {n_components} componentes.")
        return transformed_out, n_components

    def get_train_data(self, epoch, model, outs_posibilities, explained_variance=None, n_components=None):
        if self.check_filter_update_criteria(epoch):
            def batched_inspector_out(model, X, batch_size=256):
                outs = []
                n = len(X)
                for i in range(0, n, batch_size):
                    batch = X[i:i+batch_size]
                    # Convierte a tensor torch en el device correcto (por si X es numpy/tf)
                    if isinstance(batch, tf.Tensor):
                        batch = batch.numpy()
                    batch = torch.tensor(batch, dtype=torch.float32, device=next(model.parameters()).device)
                    with torch.no_grad():
                        out = model.inspector_out(batch)
                        if isinstance(out, torch.Tensor):
                            outs.append(out.cpu().numpy())
                        else:
                            outs.append(np.array(out))
                    del batch, out  # Limpieza proactiva
                return np.concatenate(outs, axis=0)
            inspector_layer_out = batched_inspector_out(model, self.X_tr, batch_size=256)
            inspector_layer_out, n_components = self.apply_pca(inspector_layer_out, explained_variance, n_components)

            original_classes = list(outs_posibilities)

            # Selección del método de clustering
            if self.filter_model == "gmm":
                print("Usando Gaussian Mixture Model (GMM) para el clustering...")
                gmm = GMM(n_components=self.y_tr.shape[1], random_state=self.random_state)
                gmm.fit(inspector_layer_out)
                clusterized_outs_proba = gmm.predict_proba(inspector_layer_out)
                clusterized_outs = clusterized_outs_proba.argmax(axis=1)
            elif self.filter_model == "smm":
                print("Usando Student Mixture Model (SMM) para el clustering...")
                smm = StudentMixture(
                    n_components=self.y_tr.shape[1],
                    random_state=self.random_state,
                    covariance_type="full",
                    max_iter=100,
                    tol=1e-3
                )
                smm.fit(inspector_layer_out)
                clusterized_outs_proba = smm.predict_proba(inspector_layer_out)
                clusterized_outs = clusterized_outs_proba.argmax(axis=1)
            else:
                raise ValueError("Método de filtrado inválido. Usa 'gmm' o 'smm'.")

            # --- ASIGNACIÓN ÓPTIMA CLASE-CLUSTER (Hungarian) ---
            num_classes = len(original_classes)
            num_clusters = clusterized_outs_proba.shape[1]

            count_matrix = np.zeros((num_classes, num_clusters), dtype=int)
            clases_true = self.y_tr.argmax(axis=1)
            for i in range(len(clases_true)):
                count_matrix[clases_true[i], clusterized_outs[i]] += 1

            row_ind, col_ind = linear_sum_assignment(-count_matrix)
            class_cluster_to_real = {col: original_classes[row] for row, col in zip(row_ind, col_ind)}

            print("Asignación clase-cluster:")
            print(class_cluster_to_real)

            # Reordena las probabilidades según asignación óptima
            clases_true = self.y_tr.argmax(axis=1)
            probs = []
            for i in range(len(clases_true)):
                clase_real = clases_true[i]
                cluster_asociado = [c for c, clase in class_cluster_to_real.items() if clase == clase_real]
                if len(cluster_asociado) == 0:
                    probs.append(0.0)
                else:
                    probs.append(clusterized_outs_proba[i, cluster_asociado[0]])
            prob_correct_class_cluster = np.array(probs)

            size_set_train = self.X_tr.shape[0]
            print(f"Tamaño del set de entrenamiento: {size_set_train}")

            # === AQUÍ VA LA LÓGICA LIMPIA DEL UMBRAL ===
            filtered_indices_per_class = []
            min_pureza = 0.6    # No filtrar si el cluster de la clase es poco puro
            min_keep = 0.5      # Siempre dejar al menos 50% de datos de la clase
            for class_it in original_classes:
                # ¿Cuál es el cluster asociado a esta clase?
                cluster_asociado = [c for c, clase in class_cluster_to_real.items() if clase == class_it]
                if not cluster_asociado:
                    print(f"[WARNING] Clase {class_it} no tiene cluster asignado. No se filtra nada.")
                    continue
                cluster_idx = cluster_asociado[0]
                # Pureza del cluster para esta clase
                in_cluster = (clusterized_outs == cluster_idx)
                if in_cluster.sum() == 0:
                    print(f"[WARNING] Cluster {cluster_idx} está vacío. No se filtra nada para la clase {class_it}.")
                    continue
                clases_in_cluster = clases_true[in_cluster]
                pureza = (clases_in_cluster == class_it).sum() / in_cluster.sum()
                print(f"Pureza del cluster {cluster_idx} para la clase {class_it}: {pureza:.2f}")
                class_mask = (clases_true == class_it)
                class_probs = np.round(prob_correct_class_cluster[class_mask], 2)
                n_total = len(class_probs)
                if n_total == 0:
                    print(f"[WARNING] Clase {class_it} vacía tras clusterización.")
                    continue
                # Si la pureza del cluster es baja, no filtres
                if pureza < min_pureza:
                    print(f"[PROTEGIDO] Clase {class_it} NO filtrada (pureza < {min_pureza})")
                    indices_keep = np.where(class_mask)[0]
                else:
                    threshold = np.percentile(class_probs, self.filter_percentile * 100)
                    threshold = round(threshold, 2)
                    indices_above_threshold = np.where(class_mask)[0][class_probs >= threshold]
                    # Protección mínima: no filtres más del X% de la clase
                    if len(indices_above_threshold) < int(min_keep * n_total):
                        print(f"[PROTEGIDO] No se filtra más del {min_keep*100:.0f}% de la clase {class_it}.")
                        num_keep = int(min_keep * n_total)
                        sorted_indices = np.argsort(-class_probs)   # mayor prob a menor
                        indices_keep = np.where(class_mask)[0][sorted_indices[:num_keep]]
                    else:
                        indices_keep = indices_above_threshold
                    print(f'Filtrando {n_total - len(indices_keep)} de {n_total} elementos en clase {class_it}')
                filtered_indices_per_class.append(indices_keep)

            # ===========================================

            self.filtered_index = np.concatenate(filtered_indices_per_class)
            original_indices = np.arange(self.X_tr.shape[0], dtype=int)
            removed_data_indices = list(set(original_indices).difference(set(self.filtered_index)))
            removed_original_indices = self.original_indices[removed_data_indices]

            self.removed_data_indices = removed_original_indices.tolist()
            self.all_removed_indices.extend(removed_original_indices.tolist())
            print(f"Datos removidos: {removed_original_indices}")

            if len(self.filtered_index) == 0:
                print("No se identificaron outliers, se utiliza el dataset filtrado previamente")
                return self.previous_X_tr, self.previous_y_tr, self.original_indices, self.all_removed_indices, self.inspector_layer_out

            filtered_X_tr = tf.gather(self.X_tr, self.filtered_index)
            filtered_y_tr = tf.gather(self.y_tr, self.filtered_index).numpy()
            filtered_original_indices = self.original_indices[self.filtered_index]
            size_set_post = filtered_X_tr.shape[0]
            print("El dataset ha sido filtrado.")
            print(f"Tamaño de datos removidos: {size_set_train - size_set_post}")

            if self.train_with_outliers:
                removed_data = tf.gather(self.X_tr, np.array(removed_data_indices, dtype=int))
                removed_labels = tf.gather(self.y_tr, np.array(removed_data_indices, dtype=int))
                removed_indices = tf.gather(self.original_indices, np.array(removed_data_indices, dtype=int))
                num_removed = 3 * len(removed_data_indices)
                if num_removed == 0:
                    print("No hay datos para remover como outliers, se utiliza el dataset filtrado previo")
                    return self.previous_X_tr, self.previous_y_tr, self.original_indices, self.all_removed_indices, self.inspector_layer_out
                all_indices = set(range(len(self.X_tr)))
                excluded_indices = set(removed_data_indices)
                available_indices = list(all_indices - excluded_indices)
                random_indices = np.random.choice(available_indices, num_removed, replace=False)
                random_data = tf.gather(self.X_tr, random_indices)
                random_labels = tf.gather(self.y_tr, random_indices)
                random_original_indices = np.array(self.original_indices)[random_indices]
                filtered_X_tr = np.concatenate((removed_data, random_data), axis=0)
                filtered_y_tr = np.concatenate((removed_labels, random_labels), axis=0)
                filtered_original_indices = np.concatenate((removed_indices, random_original_indices), axis=0)
                print(f"Entrenamiento con outliers: se agregaron {num_removed} puntos removidos y {num_removed} puntos aleatorios.")

            self.X_tr = filtered_X_tr
            self.y_tr = filtered_y_tr
            self.original_indices = filtered_original_indices
            self.previous_X_tr = self.X_tr
            self.previous_y_tr = self.y_tr
            self.inspector_layer_out = inspector_layer_out

        return self.return_filtered_data()


    def return_filtered_data(self):
        return self.X_tr, self.y_tr, self.original_indices, self.all_removed_indices, self.inspector_layer_out

