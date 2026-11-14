"""Three fixed Hybrid candidates; all fitting is restricted to supplied Train."""
import importlib.metadata
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
import model as M

SEEDS = (42, 43, 44)
# Candidate settings are fixed before measuring any new CV/Public results.
RECIPES = {
    "hybrid_control": ("current", "current"),
    "temporal_tree": ("current", "all"),
    "temporal_both": ("all", "all"),
}


def metrics(target, probabilities):
    prediction = probabilities.argmax(1)
    return {
        "macro_f1":float(f1_score(target, prediction, labels=list(range(6)), average="macro", zero_division=0)),
        "class_f1":f1_score(target, prediction, labels=list(range(6)), average=None, zero_division=0).tolist(),
        "confusion_matrix":confusion_matrix(target, prediction, labels=list(range(6))).tolist(),
    }


def package_versions():
    return {name:importlib.metadata.version(name) for name in
            ("torch", "torchvision", "numpy", "pandas", "scikit-learn", "pillow", "joblib")}


def train_heads(images, sensors, target, device, epochs=16):
    states = []
    for seed in SEEDS:
        M.seed_all(seed)
        net = M.Fusion(sensors.shape[1]).to(device)
        loader = DataLoader(TensorDataset(images, torch.from_numpy(sensors), torch.from_numpy(target)),
                            batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
        criterion = nn.CrossEntropyLoss()
        for _ in range(epochs):
            net.train()
            for image, sensor, y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(net(image.to(device), sensor.to(device)), y.to(device))
                loss.backward()
                optimizer.step()
        states.append({name:value.detach().cpu().clone() for name,value in net.state_dict().items()})
        del net, optimizer
    return states


def fit_family(frame, raw_images, device, feature_tables=None, epochs=16, tree_iterations=250, pca_components=32, recipes=None):
    if epochs < 1 or tree_iterations < 1 or pca_components < 1:
        raise ValueError("Training parameters must be positive")
    target = M.labels(frame)
    if set(target) != set(range(6)):
        raise ValueError("Training split must contain all six classes")
    if tuple(raw_images.shape) != (len(frame), 512) or not torch.isfinite(raw_images).all():
        raise ValueError("Invalid image embeddings")
    recipes = list(RECIPES) if recipes is None else list(recipes)
    if not recipes or len(set(recipes)) != len(recipes) or any(name not in RECIPES for name in recipes):
        raise ValueError("Select distinct supported recipes")
    needed_mlp = {RECIPES[name][0] for name in recipes}
    needed_tree = {RECIPES[name][1] for name in recipes}
    needed_features = needed_mlp | needed_tree
    tables = feature_tables or {name:M.build_sensor_features(frame, name) for name in needed_features}
    for name,table in tables.items():
        if len(table) != len(frame) or list(table.columns) != M.feature_names(name):
            raise ValueError("Feature table and training schema differ")
    prep = {name:M.fit_preprocessing(table) for name,table in tables.items()}
    sensors = {name:M.apply_preprocessing(table, prep[name]) for name,table in tables.items()}
    mean = raw_images.mean(0)
    std = raw_images.std(0, unbiased=False).clamp_min(1e-6)
    images = (raw_images-mean)/std
    pca = PCA(n_components=min(pca_components, len(frame)-1, 512), svd_solver="randomized", random_state=42)
    # Keep the same PCA fitting/projection convention as the original Hybrid control.
    reduced = pca.fit_transform(images.numpy())
    states, classifiers = {}, {}
    for name in sorted(needed_mlp):
        print("Training fixed-epoch three-seed Fusion:", name, flush=True)
        states[name] = train_heads(images, sensors[name], target, device, epochs)
    for name in sorted(needed_tree):
        tree = HistGradientBoostingClassifier(learning_rate=.06, max_iter=tree_iterations,
            max_leaf_nodes=15, min_samples_leaf=32, l2_regularization=5., early_stopping=False, random_state=42)
        tree.fit(np.column_stack([sensors[name], reduced]), target)
        if not np.array_equal(tree.classes_, np.arange(6)):
            raise ValueError("Unexpected HGB class order")
        classifiers[name] = tree
    bundles = {}
    for recipe in recipes:
        mlp_features, tree_features = RECIPES[recipe]
        used = set([mlp_features, tree_features])
        bundles[recipe] = {
            "format_version":3, "model_family":"hybrid_v2_temporal", "recipe":recipe,
            "mlp_features":mlp_features, "tree_features":tree_features,
            "schemas":{name:M.feature_names(name) for name in used},
            "preprocessing":{name:{k:v.tolist() for k,v in prep[name].items()} for name in used},
            "states":states[mlp_features], "seeds":list(SEEDS), "epochs":epochs,
            "image_mean":mean, "image_std":std, "mixture_weights":[.5, .5],
            "class_labels":[1, 2, 3, 4, 5, 6], "tree_file":"tree.joblib",
            "environment_versions":package_versions(),
        }
    trees = {recipe:{"pca":pca, "hgb":classifiers[RECIPES[recipe][1]]} for recipe in recipes}
    return bundles, trees


@torch.inference_mode()
def mlp_probability(images, sensors, states, device):
    probabilities = []
    for state in states:
        net = M.Fusion(sensors.shape[1]).to(device)
        net.load_state_dict(state)
        net.eval()
        chunks = []
        for start in range(0, len(sensors), 256):
            logits = net(images[start:start+256].to(device), torch.from_numpy(sensors[start:start+256]).to(device))
            chunks.append(logits.softmax(1).cpu().numpy())
        probabilities.append(np.concatenate(chunks).astype(np.float64))
        del net
    return np.mean(probabilities, axis=0)


@torch.inference_mode()
def predict(frame, raw_images, bundle, trees, device, feature_tables=None, return_components=False):
    if bundle.get("format_version") != 3 or bundle.get("model_family") != "hybrid_v2_temporal":
        raise ValueError("Unsupported checkpoint; keep original model code for original weights")
    recipe = bundle.get("recipe")
    if recipe not in RECIPES or tuple([bundle["mlp_features"], bundle["tree_features"]]) != RECIPES[recipe]:
        raise ValueError("Checkpoint recipe differs from code")
    if bundle["class_labels"] != [1, 2, 3, 4, 5, 6] or bundle["mixture_weights"] != [.5, .5]:
        raise ValueError("Unsupported class mapping or mixture")
    if len(bundle["states"]) != len(SEEDS) or tuple(bundle["seeds"]) != SEEDS:
        raise ValueError("Unexpected ensemble members")
    used = set([bundle["mlp_features"], bundle["tree_features"]])
    tables = feature_tables or {name:M.build_sensor_features(frame, name) for name in used}
    sensors = {}
    for name in used:
        if bundle["schemas"][name] != M.feature_names(name) or list(tables[name].columns) != bundle["schemas"][name] or len(tables[name]) != len(frame):
            raise ValueError("Checkpoint feature schema mismatch")
        sensors[name] = M.apply_preprocessing(tables[name], bundle["preprocessing"][name])
    if tuple(raw_images.shape) != (len(frame), 512) or not torch.isfinite(raw_images).all():
        raise ValueError("Invalid inference image features")
    images = (raw_images-bundle["image_mean"])/bundle["image_std"]
    mlp = mlp_probability(images, sensors[bundle["mlp_features"]], bundle["states"], device)
    if not np.array_equal(trees["hgb"].classes_, np.arange(6)):
        raise ValueError("Tree class mapping differs from checkpoint")
    reduced = trees["pca"].transform(images.numpy())
    tree_probability = trees["hgb"].predict_proba(np.column_stack([sensors[bundle["tree_features"]], reduced]))
    probability = .5*mlp + .5*tree_probability
    for values in (mlp, tree_probability, probability):
        if values.shape != (len(frame), 6) or not np.isfinite(values).all() or (values < 0).any() or not np.allclose(values.sum(1), 1., atol=1e-5):
            raise ValueError("Invalid class probabilities")
    return {"hybrid":probability, "mlp":mlp, "tree":tree_probability} if return_components else probability
