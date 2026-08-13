"""Published external prototype-based classifiers implemented from scratch (numpy/torch).
All classifiers share the repository metric convention:
  accuracy   = correct confident predictions / confident predictions
  coverage   = confident predictions / total test points
  compression = number of stored representatives / training set size

Classes
  LVQ1            (Kohonen 1990)                                  -> Kohonen, Self-Organizing Maps, 1990
  GLVQ            (Sato & Yamada 1995)                            -> Sato & Yamada, "Generalized LVQ", 1995
  PrototypicalNet (Snell et al. 2017, learned linear embedding)  -> Snell et al., "Prototypical Networks", 2017
  LRVQ            (LVQ + local reject option, Fischer et al.)     -> Fischer et al., "LVQ with rejection", 2016
"""
import numpy as np


def _evaluate(pred, conf, is_conf, y_true, n_reprs, train_size):
    is_conf = is_conf.astype(bool)
    total = len(y_true)
    acc = float(np.mean(pred[is_conf] == y_true[is_conf])) if is_conf.any() else 0.0
    cov = float(is_conf.mean()) if total else 0.0
    return dict(accuracy=acc, coverage=cov, compression=n_reprs / train_size,
                n_reprs=int(n_reprs), confident=int(is_conf.sum()), total=total)


class LVQ1:
    """Learning Vector Quantization (LVQ1), nearest-prototype rule (Kohonen 1990)."""

    def __init__(self, m=8, epochs=25, lr=0.05, lr_decay=0.99, seed=0, **kw):
        self.m = m
        self.epochs = epochs
        self.lr0 = lr
        self.lr_decay = lr_decay
        self.seed = seed
        self.prototypes = None
        self.protolabels = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        classes = np.unique(y)
        protos, plabels = [], []
        for c in classes:
            pts = X[y == c]
            idx = rng.choice(len(pts), size=min(self.m, len(pts)), replace=False)
            protos.append(pts[idx])
            plabels.append(np.repeat(c, len(idx)))
        self.prototypes = np.vstack(protos).astype(np.float64)
        self.protolabels = np.concatenate(plabels)
        train_size = len(X)
        order = np.arange(train_size)
        for epoch in range(self.epochs):
            rng.shuffle(order)
            lr = self.lr0 * (self.lr_decay ** epoch)
            for i in order:
                d = np.sum((self.prototypes - X[i]) ** 2, axis=1)
                j = int(np.argmin(d))
                if self.protolabels[j] == y[i]:
                    self.prototypes[j] += lr * (X[i] - self.prototypes[j])
                else:
                    self.prototypes[j] -= lr * (X[i] - self.prototypes[j])
        return self

    def predict(self, X):
        d = np.sum((X[:, None, :] - self.prototypes[None, :, :]) ** 2, axis=2)
        j = np.argmin(d, axis=1)
        pred = self.protolabels[j].astype(int)
        conf = np.ones(len(X))
        is_conf = np.ones(len(X), dtype=bool)
        return pred, conf, is_conf, len(self.prototypes)

    def evaluate(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        pred, conf, is_conf, n = self.predict(X_test)
        return _evaluate(pred, conf, is_conf, y_test, n, len(X_train))


class GLVQ:
    """Generalized LVQ (Sato & Yamada 1995): relative squared-distance update."""

    def __init__(self, m=8, epochs=30, lr=0.01, gamma=1.0, seed=0, **kw):
        self.m = m
        self.epochs = epochs
        self.lr = lr
        self.gamma = gamma
        self.seed = seed
        self.prototypes = None
        self.protolabels = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        classes = np.unique(y)
        protos, plabels = [], []
        for c in classes:
            pts = X[y == c]
            idx = rng.choice(len(pts), size=min(self.m, len(pts)), replace=False)
            protos.append(pts[idx])
            plabels.append(np.repeat(c, len(idx)))
        self.prototypes = np.vstack(protos).astype(np.float64)
        self.protolabels = np.concatenate(plabels)
        order = np.arange(len(X))
        for epoch in range(self.epochs):
            rng.shuffle(order)
            lr = self.lr * (0.95 ** epoch)
            for i in order:
                d = np.sum((self.prototypes - X[i]) ** 2, axis=1)
                same = np.where(self.protolabels == y[i])[0]
                diff = np.where(self.protolabels != y[i])[0]
                if same.size == 0 or diff.size == 0:
                    continue
                jp = int(same[np.argmin(d[same])])
                jn = int(diff[np.argmin(d[diff])])
                dp = d[jp]
                dn = d[jn]
                denom = (dp + dn) ** 2
                # Biehl et al. GLVQ update (squared distances)
                self.prototypes[jp] += lr * (2 * dn / denom) * (X[i] - self.prototypes[jp])
                self.prototypes[jn] -= lr * (2 * dp / denom) * (X[i] - self.prototypes[jn])
        return self

    def predict(self, X):
        d = np.sum((X[:, None, :] - self.prototypes[None, :, :]) ** 2, axis=2)
        j = np.argmin(d, axis=1)
        pred = self.protolabels[j].astype(int)
        conf = np.ones(len(X))
        is_conf = np.ones(len(X), dtype=bool)
        return pred, conf, is_conf, len(self.prototypes)

    def evaluate(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        pred, conf, is_conf, n = self.predict(X_test)
        return _evaluate(pred, conf, is_conf, y_test, n, len(X_train))


class SklearnGLVQ:
    """Open-source GLVQ via sklearn-lvq (BSD-3, Biehl/Schneider/Bunte), with
    a documented compatibility shim for the current stack (scipy>=1.8 requires a
    1-D x0; numpy>=2 removed np.math). Behaves like GLVQ (no reject option):
    confidence = 1, full coverage."""

    def __init__(self, m=1, max_iter=2500, gtol=1e-5, seed=0, **kw):
        self.m = m
        self.max_iter = max_iter
        self.gtol = gtol
        self.seed = seed

    def fit(self, X, y):
        import numpy as np
        classes = np.unique(np.asarray(y))
        if len(classes) == 1:
            # single-class training set: GLVQ cannot fit, predict the only class
            self._single_class_ = int(classes[0])
            self._is_single_class_ = True
            return self
        self._is_single_class_ = False
        from sklearn_lvq.glvq import GlvqModel
        import scipy.optimize as so

        class _Compat(GlvqModel):
            def phi(self, x):
                x = np.clip(x, -500.0, 500.0)
                return 1.0 / (1.0 + np.exp(-self.beta * x))

            def phi_prime(self, x):
                x = np.clip(x, -500.0, 500.0)
                e = np.exp(-self.beta * x)
                return self.beta * e / (1.0 + e) ** 2

            def _optimize(self, x, y, random_state):
                label_equals_prototype = y[np.newaxis].T == self.c_w_
                res = so.minimize(
                    fun=lambda vs: self._optfun(variables=vs, training_data=x,
                                                label_equals_prototype=label_equals_prototype),
                    jac=lambda vs: self._optgrad(variables=vs, training_data=x,
                                                 label_equals_prototype=label_equals_prototype,
                                                 random_state=random_state),
                    method='l-bfgs-b', x0=self.w_.ravel(),
                    options={'gtol': self.gtol, 'maxiter': self.max_iter})
                self.w_ = res.x.reshape(self.w_.shape)
                self.n_iter_ = res.nit

        self.model_ = _Compat(prototypes_per_class=self.m, random_state=self.seed,
                              max_iter=self.max_iter, gtol=self.gtol)
        self.model_.fit(np.asarray(X, dtype=np.float64), np.asarray(y))
        return self

    def predict(self, X):
        if getattr(self, '_is_single_class_', False):
            n = len(X)
            pred = np.full(n, self._single_class_, dtype=int)
            conf = np.ones(n)
            is_conf = np.ones(n, dtype=bool)
            return pred, conf, is_conf, 1
        pred = self.model_.predict(np.asarray(X, dtype=np.float64)).astype(int)
        conf = np.ones(len(X))
        is_conf = np.ones(len(X), dtype=bool)
        return pred, conf, is_conf, self.model_.w_.shape[0]

    def evaluate(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        pred, conf, is_conf, n = self.predict(X_test)
        return _evaluate(pred, conf, is_conf, y_test, n, len(X_train))


class LRVQ:
    """LVQ1 with a local reject option: each prototype owns a ball of radius
    = distance to nearest prototype of another class * margin. Points that do
    not fall into any ball are deferred (no confident prediction). The margin
    is auto-tuned on the training set so that the confident-prediction
    coverage lands close to `target_coverage` (balanced against the P epsilon N
    operating point used in the paper)."""

    def __init__(self, m=8, epochs=25, lr=0.05, margin=0.95, target_coverage=None,
                 grid=None, seed=0, **kw):
        self.m = m
        self.epochs = epochs
        self.lr0 = lr
        self.margin = margin
        self.target_coverage = target_coverage
        self.grid = grid if grid is not None else np.arange(0.05, 1.0, 0.05)
        self.seed = seed
        self._lvq = LVQ1(m=m, epochs=epochs, lr=lr, seed=seed)
        self.radii = None
        self.best_margin = None

    def fit(self, X, y, margin=None):
        self._lvq.fit(X, y)
        P = self._lvq.prototypes
        L = self._lvq.protolabels
        radii_full = np.zeros(len(P))
        for j in range(len(P)):
            d = np.sum((P - P[j]) ** 2, axis=1)
            other = np.where(L != L[j])[0]
            radii_full[j] = np.sqrt(np.min(d[other])) if other.size else np.inf
        if margin is not None:
            self.best_margin = margin
            self.radii = radii_full * margin
        else:
            d_tr = np.sqrt(np.sum((X[:, None, :] - P[None, :, :]) ** 2, axis=2))
            jn = np.argmin(d_tr, axis=1)
            other = ~(L[None, :] == L[jn[:, None]])
            dist_other = np.where(other, d_tr, np.inf).min(axis=1)
            self.best_margin = None
            best_abs = np.inf
            for m in self.grid:
                cov = float(np.mean(d_tr[np.arange(len(X)), jn] / np.maximum(dist_other, 1e-12) <= m))
                if self.target_coverage is None:
                    self.best_margin = self.margin
                    break
                if abs(cov - self.target_coverage) < best_abs:
                    best_abs = abs(cov - self.target_coverage)
                    self.best_margin = float(m)
            self.radii = radii_full * self.best_margin
        return self

    def predict(self, X):
        d = np.sqrt(np.sum((X[:, None, :] - self._lvq.prototypes[None, :, :]) ** 2, axis=2))
        j = np.argmin(d, axis=1)
        within = d[np.arange(len(X)), j] <= self.radii[j]
        is_conf = within.astype(bool)
        j = np.where(is_conf, j, -1)
        pred = np.array([self._lvq.protolabels[jj] if jj >= 0 else -1 for jj in j], dtype=int)
        conf_all = np.zeros(len(X))
        conf_all[is_conf] = 1.0 - d[np.arange(len(X))[is_conf], j[is_conf]] / self.radii[j[is_conf]]
        return pred, conf_all, is_conf, len(self._lvq.prototypes)

    def evaluate(self, X_train, y_train, X_test, y_test):
        margin = None if self.target_coverage is not None else self.margin
        self.fit(X_train, y_train, margin=margin)
        pred, conf, is_conf, n = self.predict(X_test)
        return _evaluate(pred, conf, is_conf, y_test, n, len(X_train))


class PrototypicalNet:
    """Prototypical Network with a learned linear embedding (Snell et al. 2017).
    One prototype per class (mean of embedded support). Compression = n_classes.
    """

    def __init__(self, dim_out=None, epochs=15, lr=0.01, batch=256, seed=0, n_support=5, **kw):
        self.dim_out = dim_out
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.seed = seed
        self.n_support = n_support
        self.W = None
        self.means = None
        self.classes = None

    def fit(self, X, y):
        import torch
        import numpy as np
        classes = np.unique(np.asarray(y))
        if len(classes) == 1:
            # single-class training set: ProtoNet cannot fit, predict the only class
            self._is_single_class_ = True
            self._single_class_ = int(classes[0])
            return self
        self._is_single_class_ = False
        torch.manual_seed(self.seed)
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32))
        yt = torch.as_tensor(np.asarray(y, dtype=np.int64))
        d = Xt.shape[1]
        out_dim = self.dim_out or d
        W = torch.randn(d, out_dim, dtype=torch.float32) * (2.0 / np.sqrt(d))
        W.requires_grad_(True)
        opt = torch.optim.Adam([W], lr=self.lr)
        classes = torch.unique(yt)
        self.classes = classes.numpy()
        n = len(yt)
        n_support = self.n_support
        if n_support is None:
            n_support = min(50, int((yt == classes[0]).sum()), int((yt == classes[-1]).sum()))
        n_support = int(n_support)
        for ep in range(self.epochs):
            opt.zero_grad()
            # episode support: n_support random support points per class
            support_idx, support_y = [], []
            for c in classes:
                idx_c = torch.nonzero(yt == c).flatten()
                ns = min(int(n_support), len(idx_c))
                perm_c = idx_c[torch.randperm(len(idx_c))[:ns]]
                support_idx.append(perm_c)
                support_y.append(torch.full((ns,), c.item(), dtype=torch.int64))
            support_idx = torch.cat(support_idx)
            support_y = torch.cat(support_y)
            e_s = Xt[support_idx] @ W
            means = {c.item(): e_s[support_y == c].mean(0) for c in classes}
            # query batch
            perm = torch.randperm(n)[: self.batch]
            xb = Xt[perm]
            yb = yt[perm]
            e_q = xb @ W
            dists = torch.stack([torch.sum((e_q - m.unsqueeze(0)) ** 2, dim=1) for m in means.values()], dim=1)
            logits = -dists
            logp = torch.log_softmax(logits, dim=1)
            loss = torch.nn.functional.nll_loss(logp, yb)
            loss.backward()
            opt.step()
        self.W = W.detach().numpy()
        e_train = np.asarray(X, dtype=np.float32) @ self.W
        self.means = np.stack([e_train[y == c].mean(0) for c in self.classes])
        return self

    def predict(self, X):
        if getattr(self, '_is_single_class_', False):
            n = len(X)
            pred = np.full(n, self._single_class_, dtype=int)
            conf = np.ones(n)
            is_conf = np.ones(n, dtype=bool)
            return pred, conf, is_conf, 1
        e = np.asarray(X, dtype=np.float32) @ self.W
        d = np.sum((e[:, None, :] - self.means[None, :, :]) ** 2, axis=2)
        j = np.argmin(d, axis=1)
        pred = self.classes[j].astype(int)
        conf = np.ones(len(X))
        is_conf = np.ones(len(X), dtype=bool)
        return pred, conf, is_conf, len(self.classes)

    def evaluate(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        pred, conf, is_conf, n = self.predict(X_test)
        return _evaluate(pred, conf, is_conf, y_test, n, len(X_train))
