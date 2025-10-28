import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
import tensorflow as tf
from scipy import linalg, special
from scipy.stats import chi2, f as f_dist
import torch
import warnings

# =========================
# Util: clip [0,1] con warning
# =========================
def _clip01_with_warning(name, value, allow_none=False):
    if allow_none and value is None:
        return None
    try:
        v = float(value)
    except Exception:
        warnings.warn(f"'{name}' no es numerico ({value}); intentando convertir con float().", UserWarning)
        v = float(value)
    v_clipped = float(np.clip(v, 0.0, 1.0))
    if v != v_clipped:
        dir_txt = "mayor que 1" if v > 1.0 else "menor que 0"
        warnings.warn(f"'{name}' {dir_txt}: recibido {v}, se ajusta a {v_clipped}.", UserWarning)
    return v_clipped


# =========================
# StudentMixture (t-Student Mixture)
# =========================
class StudentMixture:
    """Modelo de mezcla de distribuciones t de Student."""
    def __init__(self, n_components, covariance_type='full', tol=1e-3, reg_covar=1e-6,
                 max_iter=100, random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.random_state = np.random.RandomState(random_state)
        self.converged_ = False

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        self.means_ = X[self.random_state.choice(n_samples, self.n_components, replace=False)]
        cov_global = np.cov(X.T) + self.reg_covar * np.eye(n_features)
        self.covariances_ = np.array([cov_global.copy() for _ in range(self.n_components)])
        self.degrees_of_freedom_ = np.full(self.n_components, 10.0)

    def _estimate_log_prob(self, X):
        n_samples, n_features = X.shape
        log_prob = np.empty((n_samples, self.n_components))
        for k in range(self.n_components):
            nu = self.degrees_of_freedom_[k]
            if nu <= 0:
                raise ValueError(f"Grados de libertad no validos: {nu}. Deben ser > 0.")
            diff = X - self.means_[k]
            precision = linalg.inv(self.covariances_[k])
            quad_form = np.sum(diff @ precision * diff, axis=1)
            log_det_cov = np.log(max(linalg.det(self.covariances_[k]), 1e-10))
            log_prob[:, k] = (
                special.gammaln((nu + n_features) / 2)
                - special.gammaln(nu / 2)
                - 0.5 * (n_features * np.log(nu * np.pi) + log_det_cov)
                - 0.5 * (nu + n_features) * np.log(1 + np.maximum(quad_form / nu, 1e-10))
            )
        return log_prob

    def _e_step(self, X):
        log_prob = self._estimate_log_prob(X)
        weighted_log_prob = log_prob + np.log(self.weights_)
        log_prob_norm = special.logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return np.mean(log_prob_norm), log_resp

    def _m_step(self, X, log_resp):
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.weights_ = nk / nk.sum()
        self.means_ = np.dot(resp.T, X) / nk[:, np.newaxis]
        self.covariances_ = self._estimate_covariances(X, resp, nk)
        self.degrees_of_freedom_ = self._update_degrees_of_freedom(X, resp, nk)

    def _estimate_covariances(self, X, resp, nk):
        n_samples, n_features = X.shape
        covariances = np.empty((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_diff = resp[:, k][:, np.newaxis] * diff
            covariances[k] = np.dot(weighted_diff.T, diff) / max(nk[k], 1e-10) + self.reg_covar * np.eye(n_features)
        return covariances

    def _update_degrees_of_freedom(self, X, resp, nk):
        n_samples, n_features = X.shape
        new_dof = np.empty(self.n_components)
        for k in range(self.n_components):
            diff = X - self.means_[k]
            quad_form = np.sum(diff @ linalg.inv(self.covariances_[k]) * diff, axis=1)
            weighted_quad_form = np.dot(resp[:, k], quad_form)
            new_dof[k] = max(2 * (n_features + nk[k]) / (nk[k] - weighted_quad_form / (self.degrees_of_freedom_[k] + 2)), 1.0)
        return new_dof

    def fit(self, X):
        self._initialize_parameters(X)
        prev_w = self.weights_.copy
        for n_iter in range(self.max_iter):
            _, log_resp = self._e_step(X)
            self._m_step(X, log_resp)
            if np.allclose(self.weights_, self.weights_ if prev_w is None else prev_w(), atol=self.tol):
                self.converged_ = True
                print(f"Convergencia alcanzada en la iteracion {n_iter}.")
                break
            prev_w = self.weights_.copy

    def predict_proba(self, X):
        _, log_resp = self._e_step(X)
        return np.exp(log_resp)


# =========================
# Helpers de tipo
# =========================
def _to_numpy(x):
    if tf.is_tensor(x):
        return x.numpy()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _take(arr, idx):
    """Indexa arr segun idx respetando su tipo."""
    if tf.is_tensor(arr):
        return tf.gather(arr, idx)
    if isinstance(arr, torch.Tensor):
        idx_t = torch.as_tensor(idx, device=arr.device)
        return arr.index_select(0, idx_t)
    return np.asarray(arr)[idx]

def _maybe_to_numpy(x):
    """Convierte a numpy solo si no lo es ya."""
    if isinstance(x, np.ndarray):
        return x
    return _to_numpy(x)


# =========================
# DataSelector (GMM/SMM) + score compuesto y filtrado SOFT + reduccion (PCA)
# =========================
class DataSelector:
    def __init__(self,
                 X_tr, y_tr,
                 epochs_to_start_filter,
                 update_period_in_epochs,
                 filter_percentile=0.10,            # fraccion a eliminar por clase (0..1), bottom-p%
                 random_state=None,
                 train_with_outliers=False,
                 filter_model="gmm",                # 'gmm' o 'smm'
                 gating_score_abs=0.90,             # gating absoluto (sobre S compuesto). None para desactivar
                 drop_below_score=0.50,             # floor absoluto (sobre S compuesto)
                 tau=1.0,                           # temperatura sobre R
                 batch_size_forward=256,
                 gmm_covariance_type='full',
                 gmm_reg_covar=1e-6,
                 weight_score=0.5,                  # w1 (consistencia clase–cluster)
                 weight_maha=0.5):                  # w2 (cola geométrica por clase)
        self.X_tr = X_tr
        self.y_tr = y_tr.numpy() if isinstance(y_tr, tf.Tensor) else y_tr
        self.out_clases_number = self.y_tr.shape[1]
        self.epochs_to_start_filter = int(epochs_to_start_filter)
        self.update_period_in_epochs = int(update_period_in_epochs)

        # ---- parametros del usuario con clip + warning ----
        self.filter_percentile = _clip01_with_warning("filter_percentile", filter_percentile)
        self.random_state = random_state
        self.train_with_outliers = bool(train_with_outliers)
        self.filter_model = str(filter_model).lower()

        self.gating_score_abs = _clip01_with_warning("gating_score_abs", gating_score_abs, allow_none=True)
        self.drop_below_score = _clip01_with_warning("drop_below_score", drop_below_score, allow_none=True)

        self.tau = float(tau)
        self.batch_size_forward = int(batch_size_forward)
        self.gmm_covariance_type = gmm_covariance_type
        self.gmm_reg_covar = gmm_reg_covar

        # pesos compuestos (normalizamos para que sumen 1)
        w1 = _clip01_with_warning("weight_score", weight_score)
        w2 = _clip01_with_warning("weight_maha", weight_maha)
        s = w1 + w2
        if s <= 0:
            warnings.warn("weight_score + weight_maha <= 0; se fijan ambos a 0.5.", UserWarning)
            w1, w2 = 0.5, 0.5
        else:
            w1, w2 = w1 / s, w2 / s
        self.w_score = w1
        self.w_maha  = w2

        n0 = _maybe_to_numpy(X_tr).shape[0]
        self.filtered_index = np.arange(n0)
        self.original_indices = np.arange(n0)
        self.all_removed_indices = []
        self.previous_X_tr = X_tr
        self.previous_y_tr = y_tr
        self.inspector_layer_out = []

        # ---- mapas de scores ----
        self.removal_scores = {}        # {orig_idx: S_compuesto_al_remover}
        self.last_scores_map = {}       # {orig_idx: S_compuesto_ultimo_epoch}  ← S_i
        self.last_consistency_map = {}  # {orig_idx: s_i}
        self.last_maha_tail_map = {}    # {orig_idx: s_i^M}

        # buffers del último clustering (para SMM: elegir nu_c por componente dominante)
        self._last_R = None
        self._last_mix_model = None

    # ---------- util ----------
    def check_filter_update_criteria(self, epoch):
        return (epoch >= self.epochs_to_start_filter and
                (epoch - self.epochs_to_start_filter) % self.update_period_in_epochs == 0)

    # ---------- reduccion: PCA ----------
    def apply_pca(self, Z, explained_variance=None, n_components=None):
        if explained_variance is not None:
            pca = PCA(n_components=explained_variance, random_state=self.random_state)
        elif n_components is not None:
            pca = PCA(n_components=n_components, random_state=self.random_state)
        else:
            raise ValueError("Debes proporcionar explained_variance o n_components.")
        T = pca.fit_transform(Z)
        ncomp = T.shape[1]
        if ncomp < 2:
            pca = PCA(n_components=2, random_state=self.random_state)
            T = pca.fit_transform(Z)
            ncomp = 2
        if explained_variance is not None:
            print(f"PCA realizado: se retuvo el {explained_variance*100:.1f}% de la varianza con {ncomp} componentes.")
        else:
            print(f"PCA realizado con {ncomp} componentes.")
        return T, ncomp

    # ---------- envoltorio de dimensionalidad (solo PCA) ----------
    def apply_dimensionality(self, Z, explained_variance=None, n_components=None):
        return self.apply_pca(Z, explained_variance, n_components)

    # ---------- forward del inspector ----------
    def _forward_inspector(self, model, X):
        """Soporta modelos PyTorch y TF/Keras o funciones numpy."""
        outs = []
        X_np = _maybe_to_numpy(X)
        n = len(X_np)

        # Detectar PyTorch
        is_torch_model = hasattr(model, "parameters") and callable(getattr(model, "parameters"))
        device = None
        if is_torch_model:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        for i in range(0, n, self.batch_size_forward):
            batch_np = np.asarray(X_np[i:i+self.batch_size_forward])

            if is_torch_model:
                b_t = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    o = model.inspector_out(b_t)
                o_np = _to_numpy(o)
            else:
                # Rama no-PyTorch (TF/Keras o funcion que acepte numpy)
                o = model.inspector_out(batch_np)
                o_np = _to_numpy(o)

            outs.append(o_np)

        return np.concatenate(outs, axis=0)

    # ---------- probas y score suave ----------
    @staticmethod
    def _soft_class_given_cluster(R, y_idx, C):
        N, K = R.shape
        Y = np.zeros((N, C), dtype=np.float64)
        Y[np.arange(N), y_idx] = 1.0
        Nk = R.sum(axis=0) + 1e-12           # (K,)
        Nkc = R.T @ Y                         # (K,C)
        Pc_given_k = Nkc / (Nk[:, None] + 1e-12)   # (K,C)
        Pc_given_k = np.round(Pc_given_k, 2)
        Pc_given_k = Pc_given_k / (Pc_given_k.sum(axis=1, keepdims=True) + 1e-12)
        return Pc_given_k, Nk

    @staticmethod
    def _class_scores(R, y_idx, Pc_given_k):
        Pc_for_true = Pc_given_k[:, y_idx]        # (K,N)
        scores = np.sum(R * Pc_for_true.T, axis=1)
        scores = np.clip(scores, 0.0, 1.0)
        scores = np.round(scores, 2)
        return scores  # en [0,1], redondeados

    def _cluster(self, U, K):
        """Devuelve R y el modelo de mezcla ya entrenado (GMM/SMM)."""
        if self.filter_model == "gmm":
            print("Clustering: GMM")
            model = GMM(n_components=K,
                        covariance_type=self.gmm_covariance_type,
                        reg_covar=self.gmm_reg_covar,
                        random_state=self.random_state)
            model.fit(U)
            R = model.predict_proba(U)
        elif self.filter_model == "smm":
            print("Clustering: SMM (t-Student)")
            model = StudentMixture(n_components=K, random_state=self.random_state,
                                   covariance_type="full", max_iter=100, tol=1e-3)
            model.fit(U)
            R = model.predict_proba(U)
        else:
            raise ValueError("filter_model debe ser 'gmm' o 'smm'")

        # Temperatura y normalizacion
        if self.tau != 1.0:
            R = np.power(R + 1e-12, 1.0/self.tau)
            R = R / (R.sum(axis=1, keepdims=True) + 1e-12)

        # Redondeo y normalizacion (modo soft por defecto)
        R = np.round(R, 2)
        R = R / (R.sum(axis=1, keepdims=True) + 1e-12)

        return R, model

    # ---------- Señal de Mahalanobis por clase (GMM: chi2.sf | SMM: F.sf) ----------
    def _mahalanobis_signal(self, U, y_idx, R, mix_model):
        """
        Calcula, por clase, la distancia d^2 y la señal de cola s^M_i:
          - GMM: s^M_i = chi2.sf(d^2, df=r)
          - SMM: s^M_i = F.sf(d^2 / r, r, nu_c)  (nu_c: d.o.f. del componente dominante por clase)
        Devuelve:
          - sM_all (N,) en [0,1]
          - d2_all (N,) Mahalanobis^2
          - pctl_d2 (N,) percentil de d2 dentro de su clase (0..100)
          - meta por clase
        """
        N, r = U.shape
        sM_all = np.full(N, np.nan, dtype=float)
        d2_all = np.full(N, np.nan, dtype=float)
        pctl_all = np.full(N, np.nan, dtype=float)
        meta = []  # (clase, n_c, nu_c or None)

        use_smm = (self.filter_model == "smm") and isinstance(mix_model, StudentMixture)

        classes = np.unique(y_idx)
        for c in classes:
            idx_c = np.where(y_idx == c)[0]
            n_c = idx_c.size
            if n_c == 0:
                meta.append((int(c), 0, None))
                continue

            Uc = U[idx_c]
            mu = Uc.mean(axis=0)
            cov = np.cov(Uc, rowvar=False)
            cov = cov + self.gmm_reg_covar * np.eye(r)

            try:
                cov_inv = linalg.inv(cov)
            except Exception:
                cov_inv = np.linalg.pinv(cov)

            diff = Uc - mu
            d2 = np.einsum('ij,jk,ik->i', diff, cov_inv, diff)  # (n_c,)
            d2_all[idx_c] = d2

            # percentil muestral de d2 dentro de la clase
            ranks = np.argsort(np.argsort(d2))
            pctl = 100.0 * (ranks + 1) / max(n_c, 1)
            pctl_all[idx_c] = pctl

            if use_smm and hasattr(mix_model, "degrees_of_freedom_"):
                # componente dominante por responsabilidades promedio en la clase
                R_c_mean = R[idx_c].mean(axis=0)  # (K,)
                k_star = int(np.argmax(R_c_mean))
                nu_c = float(mix_model.degrees_of_freedom_[k_star])
                nu_c = max(nu_c, 1.0)
                sM = f_dist.sf(d2 / max(r, 1), r, nu_c)
                meta.append((int(c), n_c, nu_c))
            else:
                sM = chi2.sf(d2, df=r)
                meta.append((int(c), n_c, None))

            sM_all[idx_c] = np.asarray(sM, dtype=float)

        # clip numérico y redondeo para logging estable
        sM_all = np.clip(sM_all, 0.0, 1.0)
        return sM_all, d2_all, pctl_all, meta

    def _drop_by_rank_per_class(self, comp_scores, y_idx, p, gate_abs, floor_score):
        """
        Elimina el peor p% por clase (orden ascendente de 'comp_scores' = S_i).
        Gating: si el percentil p de S_i excede 'gate_abs' → no-act en esa clase.
        Floor: dentro del bottom-p%, solo elimina si S_i < floor_score.
        Devuelve índices a CONSERVAR.
        """
        N = comp_scores.size
        kept_indices = []
        C = int(len(np.unique(y_idx)))

        for c in range(C):
            idx_c = np.where(y_idx == c)[0]
            n_c = idx_c.size
            if n_c == 0 or p <= 0.0:
                kept_indices.append(idx_c)
                print(f"[Clase {c}] n={n_c} | p={p:.3f} → no-act")
                continue

            sc = comp_scores[idx_c]
            pctl_val = float(np.percentile(sc, p*100.0))
            k_drop = int(np.floor(p * n_c))

            # gating (sobre S compuesto)
            if (gate_abs is not None) and (pctl_val > gate_abs):
                kept_indices.append(idx_c)
                print(f"[Clase {c}] gating: pctl_S={pctl_val:.3f} > {gate_abs:.3f} → no-act")
                continue

            if k_drop <= 0:
                kept_indices.append(idx_c)
                print(f"[Clase {c}] k_drop=0 (p={p:.3f}) | pctl_S={pctl_val:.3f}")
                continue

            order = np.argsort(sc)               # peores primero
            cand = idx_c[order[:k_drop]]         # candidatos a eliminar por S

            if floor_score is not None:
                cand_scores = np.round(np.clip(comp_scores[cand], 0.0, 1.0), 2)
                floor_eff = np.round(float(floor_score), 2)
                mask = cand_scores < floor_eff     # < estricto
                drop_local = set(cand[mask].tolist())
                kept_from_cand = cand[~mask]
                print(
                    f"[Clase {c}] n={n_c} | k_drop={k_drop} (p={p:.3f}) | pctl_S={pctl_val:.3f} "
                    f"| floor_S={floor_eff:.3f} → elim={np.sum(mask)} "
                    f"(se conservan {kept_from_cand.size} del bottom-p%)"
                )
            else:
                drop_local = set(cand.tolist())
                print(
                    f"[Clase {c}] n={n_c} | k_drop={k_drop} (p={p:.3f}) | pctl_S={pctl_val:.3f} "
                    f"| sin floor → elim={len(drop_local)}"
                )

            keep_local = [ii for ii in idx_c if ii not in drop_local]
            kept_indices.append(np.array(keep_local, dtype=int))

        return np.sort(np.concatenate(kept_indices)) if kept_indices else np.arange(N)

    # ---------- pipeline principal ----------
    def get_train_data(self, epoch, model, outs_posibilities, explained_variance=None, n_components=None):
        if self.check_filter_update_criteria(epoch):
            # 1) Forward del inspector
            U_inspector = self._forward_inspector(model, self.X_tr)

            # 2) Reduccion (PCA)
            U, r = self.apply_dimensionality(U_inspector, explained_variance, n_components)

            # 3) Cluster responsibilities + modelo
            K = self.y_tr.shape[1]
            R, mix_model = self._cluster(U, K)           # R:(N,K)
            self._last_R = R
            self._last_mix_model = mix_model
            y_idx = _maybe_to_numpy(self.y_tr).argmax(axis=1)

            print(f"Tamaño del set de entrenamiento (antes): {_maybe_to_numpy(self.X_tr).shape[0]}")

            # 4) Consistencia probabilistica s_i
            Pc_given_k, _ = self._soft_class_given_cluster(R, y_idx, K)
            s_prob = self._class_scores(R, y_idx, Pc_given_k)  # [0,1]
            # 5) Señal geométrica s_i^M (cola)
            s_maha, d2_all, pctl_d2, meta = self._mahalanobis_signal(U, y_idx, R, mix_model)

            # 6) Score compuesto S_i = w1*s_i + w2*s_i^M
            S_comp = np.clip(self.w_score * s_prob + self.w_maha * s_maha, 0.0, 1.0)
            S_comp = np.round(S_comp, 2)
            s_prob_r = np.round(s_prob, 2)
            s_maha_r = np.round(s_maha, 4)  # más precisión útil para colas

            # Persistir mapas (por índice original del subset)
            for i, orig in enumerate(self.original_indices):
                oi = int(orig)
                self.last_scores_map[oi] = float(S_comp[i])      # compuesto (S)
                self.last_consistency_map[oi] = float(s_prob_r[i])
                self.last_maha_tail_map[oi] = float(s_maha_r[i])

            # logging corto de la señal geométrica
            if self.filter_model == "smm":
                print("Señal geométrica (SMM, cola F): nu_c dominante por clase:")
                for c, n_c, nu_c in meta:
                    nu_txt = f"{nu_c:.2f}" if nu_c is not None else "N/A"
                    print(f"  - Clase {c}: n={n_c} | nu_c≈{nu_txt}")
            else:
                print("Señal geométrica (GMM, cola chi2).")

            # 7) Selección final por S_compuesto (gating+floor sobre S)
            kept = self._drop_by_rank_per_class(S_comp, y_idx,
                                                p=self.filter_percentile,
                                                gate_abs=self.gating_score_abs,
                                                floor_score=self.drop_below_score)

            # índices removidos (locales y originales)
            all_idx = np.arange(_maybe_to_numpy(self.X_tr).shape[0], dtype=int)
            removed_local = np.setdiff1d(all_idx, kept, assume_unique=False)
            removed_original = self.original_indices[removed_local]

            # guardar S_compuesto al momento de remover
            for loc, orig in zip(removed_local, removed_original):
                self.removal_scores[int(orig)] = float(self.last_scores_map[int(self.original_indices[loc])])

            if kept.size == 0:
                print("No se identificaron indices a conservar; mantengo dataset previo.")
                return self.previous_X_tr, self.previous_y_tr, self.original_indices, self.all_removed_indices, self.inspector_layer_out

            # 8) Actualizar dataset
            kept_sorted = np.sort(kept)
            X_new_raw = _take(self.X_tr, kept_sorted)
            y_new_raw = _take(self.y_tr, kept_sorted)
            orig_new = self.original_indices[kept_sorted]

            X_new = _maybe_to_numpy(X_new_raw)
            y_new = _maybe_to_numpy(y_new_raw)

            # acumulado de removidos (originales)
            self.all_removed_indices.extend(removed_original.tolist())
            print("El dataset ha sido filtrado.")
            print(f"Tamaño de datos removidos: {_maybe_to_numpy(self.X_tr).shape[0] - X_new.shape[0]}")

            # (opcional) entrenamiento con outliers
            if self.train_with_outliers and removed_local.size > 0:
                removed_data = _maybe_to_numpy(_take(self.X_tr, removed_local))
                removed_labels = _maybe_to_numpy(_take(self.y_tr, removed_local))
                removed_orig = self.original_indices[removed_local]

                num_removed = 3 * removed_local.size
                avail = np.setdiff1d(all_idx, removed_local, assume_unique=False)
                if avail.size > 0:
                    rand_sz = min(num_removed, avail.size)
                    rand_idx = np.random.choice(avail, size=rand_sz, replace=False)
                    rand_data = _maybe_to_numpy(_take(self.X_tr, rand_idx))
                    rand_labels = _maybe_to_numpy(_take(self.y_tr, rand_idx))
                    rand_orig = self.original_indices[rand_idx]

                    X_new = np.concatenate([removed_data, rand_data], axis=0)
                    y_new = np.concatenate([removed_labels, rand_labels], axis=0)
                    orig_new = np.concatenate([removed_orig, rand_orig], axis=0)
                    print(f"Entrenamiento con outliers: {removed_local.size} removidos + {rand_sz} aleatorios.")

            # Persistencia
            self.X_tr = X_new
            self.y_tr = y_new
            self.original_indices = orig_new
            self.previous_X_tr = self.X_tr
            self.previous_y_tr = self.y_tr
            self.inspector_layer_out = U

        return self.return_filtered_data()

    def return_filtered_data(self):
        return self.X_tr, self.y_tr, self.original_indices, self.all_removed_indices, self.inspector_layer_out

    # ---------- utilidades de scores ----------
    def get_scores_by_original_index(self):
        """
        Devuelve un dict {orig_idx: S_compuesto} que combina:
          - S_compuesto al momento de ser removido (si fue removido)
          - último S_compuesto disponible (si no fue removido)
        """
        out = dict(self.last_scores_map)
        out.update(self.removal_scores)
        return out

    def get_scores_for_indices(self, indices):
        m = self.get_scores_by_original_index()
        return np.array([m.get(int(i), np.nan) for i in indices], dtype=float)

    def get_consistency_scores_by_original_index(self):
        """Devuelve {orig_idx: s_i} (consistencia clase–cluster) del último ciclo."""
        return dict(self.last_consistency_map)

    def get_maha_tail_by_original_index(self):
        """Devuelve {orig_idx: s_i^M} (cola geométrica) del último ciclo."""
        return dict(self.last_maha_tail_map)

