from libsss import *

def optimizerr(x_train, y_train, x_val, y_val):
    X_train, y_train, X_test, y_test = x_train, y_train, x_val, y_val
    seed=42
    c = True
    while c:
        model = str(input("Model seç: "))
        n_iter = int(input("Iterations: "))
        
#################### LR Params
        if model == "LR":
            def LRobjective(params):
                print("params: ", params )
                clf = LogisticRegression(random_state = 42, **params, verbose=0, n_jobs = -1)
                clf.fit(X_train,y_train)
                score = cross_val_score(clf, X_train, y_train, cv=3, scoring='balanced_accuracy').mean()  
                print("score: ", score )
                return 1/score

            def LRoptimize(trial):
                params={"C": hp.loguniform("C", np.log(0.001), np.log(0.2)),
                        'max_iter': 1000,              #hp.choice('max_iter', np.arange(100, 1000, dtype=int)),
                        "class_weight": hp.choice("class_weight", ['balanced', None]),
                        "solver": hp.choice("solver",['newton-cg', 'lbfgs', 'sag', 'saga'])}

                best = fmin(fn = LRobjective, space = params, algo = tpe.suggest, trials = trial, max_evals = n_iter,
                            rstate = np.random.default_rng(seed))
                return best

            trial = Trials()
            LR_best = LRoptimize(trial)

            LR_best["max_iter"] = 1000                 #LR_best["max_iter"] = np.arange(100, 1000, dtype=int)[LR_best["max_iter"]]

            if LR_best["solver"] == 0:
                LR_best["solver"] = 'newton-cg'
            elif LR_best["solver"] == 1:
                LR_best["solver"] = 'lbfgs'
            elif LR_best["solver"] == 2:
                LR_best["solver"] = 'sag'
            elif LR_best["solver"] == 3:
                LR_best["solver"] = 'saga'

            if LR_best["class_weight"] == 0:
                LR_best["class_weight"] = 'balanced'
            elif LR_best["class_weight"] == 1:
                LR_best["class_weight"] = None

            LR_best['n_jobs'] = -1
            c = False

            return LR_best

#################### XGB Params
        elif model == "XGB":
            def XGBobjective(params):
                clf = XGBClassifier(**params, use_label_encoder = False, verbosity = 0, n_jobs=-1)
                clf.fit(X_train,y_train)
                score = cross_val_score(clf, X_train, y_train, cv=3, scoring='balanced_accuracy').mean()
                return 1/score

            def XGBoptimize(trial):
                params={"colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
                        "gamma": hp.uniform("gamma", 0.0, 10),
                        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
                        "max_depth": hp.choice("max_depth", np.arange(5, 30, dtype=int)),
                        "min_child_weight": hp.uniform("min_child_weight", 1, 20),
                        "n_estimators": hp.choice("n_estimators", np.arange(100, 1000, 1, dtype=int)),
                        "subsample": hp.uniform("subsample", 0.3, 1)}

                best = fmin(fn = XGBobjective, space = params, algo = tpe.suggest, trials = trial, max_evals = n_iter, 
                            rstate = np.random.default_rng(seed))
                return best

            trial = Trials()
            XGB_best = XGBoptimize(trial)

            XGB_best['use_label_encoder'] = False
            XGB_best["max_depth"] = np.arange(5, 30, dtype=int)[XGB_best["max_depth"]]
            XGB_best["n_estimators"] =  np.arange(100, 1000, 1, dtype=int)[XGB_best["n_estimators"]]
            c = False

            return XGB_best

#################### RF Params 
        elif model == "RF":
            def RFobjective(params):
                clf = RandomForestClassifier(random_state = 42, **params, n_jobs = -1)
                clf.fit(X_train,y_train)
                score = cross_val_score(clf, X_train, y_train, cv=3, scoring='balanced_accuracy').mean()
                return 1/score

            def RFoptimize(trial):
                params={"bootstrap": hp.choice("bootstrap", [True, False]),
                        'n_estimators': hp.choice('n_estimators', np.arange(100, 1000, dtype=int)),
                        "max_features": hp.choice("max_features", ["log2", "sqrt"]),
                        'max_depth':hp.choice('max_depth', np.arange(5, 30, dtype=int)),
                        'min_samples_leaf':hp.choice('min_samples_leaf', np.arange(1, 10, dtype=int)),
                        'min_samples_split':hp.choice('min_samples_split', np.arange(2, 10, dtype=int))}

                best = fmin(fn = RFobjective, space = params, algo = tpe.suggest, trials = trial, max_evals = n_iter,
                            rstate = np.random.default_rng(seed))
                return best

            trial = Trials()
            RF_best = RFoptimize(trial)

            RF_best["n_estimators"] = np.arange(100, 1000, dtype=int)[RF_best["n_estimators"]]
            RF_best["max_depth"] = np.arange(5, 30, dtype=int)[RF_best["max_depth"]]
            RF_best["min_samples_leaf"] =  np.arange(1, 10, dtype=int)[RF_best["min_samples_leaf"]]
            RF_best["min_samples_split"] =  np.arange(2, 10, dtype=int)[RF_best["min_samples_split"]]

            if RF_best["bootstrap"] == 0:
                RF_best["bootstrap"] = True
            else:
                RF_best["bootstrap"] = False

            if RF_best["max_features"] == 0:
                RF_best["max_features"] = "log2"
            else:
                RF_best["max_features"] = "sqrt"

            RF_best['n_jobs'] = -1
            c = False

            return RF_best

#################### KN Params
        elif model == "KN":
            def KNobjective(params):
                clf = KNeighborsClassifier(**params, n_jobs=-1)
                clf.fit(X_train,y_train)
                score = cross_val_score(clf, X_train, y_train, cv=3, scoring='balanced_accuracy').mean()
                return 1/score

            def KNoptimize(trial):
                params={'n_neighbors':hp.choice('n_neighbors',np.arange(2, 10, dtype=int)),
                        'weights':hp.choice('weights', ['distance', 'uniform']),
                        'leaf_size':hp.choice('leaf_size',np.arange(10, 50, dtype=int)),
                        'p':hp.choice('p',[1, 2])}

                best = fmin(fn = KNobjective, space = params, algo = tpe.suggest, trials = trial, max_evals = n_iter,
                            rstate = np.random.default_rng(seed))
                return best

            trial = Trials()
            KN_best = KNoptimize(trial)

            KN_best["n_neighbors"] = np.arange(2, 10, dtype=int)[KN_best["n_neighbors"]]
            KN_best["leaf_size"] = np.arange(10, 50, dtype=int)[KN_best["leaf_size"]]

            if KN_best["weights"] == 0:
                KN_best["weights"] = 'distance'
            else:
                KN_best["weights"] = 'uniform'

            if KN_best["p"] == 0:
                KN_best["p"] = 1
            else:
                KN_best["p"] = 2

            c = False
            return KN_best
        
#################### MLP Params   
        elif model == "MLP":
            def MLPobjective(params):
                clf = MLPClassifier(**params)
                clf.fit(X_train,y_train)
                score = cross_val_score(clf, X_train, y_train, cv=3, scoring='balanced_accuracy').mean()
                return 1/score

            def MLPoptimize(trial):
                params={'hidden_layer_sizes':(hp.choice('hidden_layer_sizes',np.arange(10, 300, dtype=int)),),
                        'activation':hp.choice('activation', ['relu', 'logistic', 'tanh']),
                        'learning_rate':hp.choice('learning_rate', ['constant', 'adaptive']),
                        "alpha": hp.uniform("alpha", 0.0001, 0.05),}

                best = fmin(fn = MLPobjective, space = params, algo = tpe.suggest, trials = trial, max_evals = n_iter,
                            rstate = np.random.default_rng(seed))
                return best

            trial = Trials()
            MLP_best = MLPoptimize(trial)

            MLP_best["hidden_layer_sizes"] = (np.arange(10, 300, dtype=int)[MLP_best["hidden_layer_sizes"]],)
            
            if MLP_best["activation"] == 0:
                MLP_best["activation"] = 'relu'
            elif MLP_best["activation"] == 1:
                MLP_best["activation"] = 'logistic'
            elif MLP_best["activation"] == 2:
                MLP_best["activation"] = 'tanh'

            if MLP_best["learning_rate"] == 0:
                MLP_best["learning_rate"] = 'constant'
            else:
                MLP_best["learning_rate"] = 'adaptive'
         
            c = False
            return MLP_best
        
        

        else: 
            print("Geçerli model seçenekleri: LR, XGB, RF, KN, MLP")
