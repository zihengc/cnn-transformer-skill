#!/usr/bin/env python3
"""Validate a cnn-transformer-pipeline config before launching a run.

Mirrors the checks in utils.validate_config, plus the ones that only show up at
runtime: unreplaced path placeholders, missing data files, task/loss mismatches,
and half-configured transformer heads. Needs only PyYAML, so it can run outside
the TensorFlow environment.

Usage:
    python scripts/check_config.py config.yaml [more.yaml ...]
    python scripts/check_config.py --repo-root /path/to/cnn-transformer-pipeline config.yaml

Exit code 0 means no errors (warnings may still be printed); 1 means errors.
"""

import argparse
import os
import sys

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")

REQUIRED = {
    'project': str,
    'train_data_paths': list,
    'train_targets': list,
    'val_data_paths': list,
    'val_targets': list,
    'batch_size': int,
    'num_epochs': int,
    'metric_pos_label': (int, str),
    'optimizer': str,
    'lr_schedule': str,
    'lr_cyc_scale_fn': str,
    'lr_init': float,
    'lr_max': float,
    'l2_reg_conv': (float, list),
    'l2_reg_dense': (float, list),
    'l2_reg_final': float,
    'lr_exp_decay_per_epoch': float,
    'lr_cyc_num_cycles': float,
    'dropout_rate_conv': (float, list),
    'dropout_rate_dense': (float, list),
    'num_conv_layers': int,
    'conv_filters': (int, list),
    'conv_width': (int, list),
    'conv_stride': (int, list),
    'max_pool_size': int,
    'max_pool_stride': int,
    'num_dense_layers': int,
    'dense_filters': (int, list),
    'shap_num_bg': int,
    'shap_num_fg': int,
}

CONV_LAYERWISE = ['conv_filters', 'conv_width', 'conv_stride',
                  'l2_reg_conv', 'dropout_rate_conv']
DENSE_LAYERWISE = ['dense_filters', 'l2_reg_dense', 'dropout_rate_dense']

REGRESSION_LOSSES = {'mean_squared_error', 'mean_absolute_error',
                     'mean_absolute_percentage_error', 'huber'}
CLASSIFICATION_LOSSES = {'sparse_categorical_crossentropy'}

PLACEHOLDER_MARKERS = ('/absolute/path/to/', '/path/to/', '<', 'CHANGEME')


class Report:
    def __init__(self, path):
        self.path = path
        self.errors = []
        self.warnings = []

    def error(self, msg):
        # The same placeholder FASTA often appears in a dozen entries; one line
        # per distinct problem keeps the output readable.
        if msg not in self.errors:
            self.errors.append(msg)

    def warn(self, msg):
        if msg not in self.warnings:
            self.warnings.append(msg)


def load(path, report):
    """Flatten the desc/value structure the way utils.get_config does."""
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        report.error("Top level of the config is not a mapping.")
        return {}
    config = {}
    for key, entry in raw.items():
        if isinstance(entry, dict) and 'value' in entry:
            config[key] = entry['value']
        else:
            report.error(f"Key '{key}' has no 'value:' field; the loader would drop it.")
    return config


def check_required(config, report):
    for key, types in REQUIRED.items():
        if key not in config:
            report.error(f"Missing required key: {key}")
            continue
        if not isinstance(types, tuple):
            types = (types,)
        value = config[key]
        # bool is an int subclass; an int key set to true is a mistake worth catching.
        if isinstance(value, bool) and bool not in types:
            report.error(f"Key '{key}' is a boolean; expected {_names(types)}.")
            continue
        if float in types and isinstance(value, int):
            continue  # an integer is an acceptable float
        if not isinstance(value, types):
            hint = ""
            if float in types and isinstance(value, str):
                hint = (" YAML 1.1 needs a dot in exponent notation:"
                        " write 4.e-4 rather than 4e-4.")
            report.error(f"Key '{key}' is {type(value).__name__};"
                         f" expected {_names(types)}.{hint}")


def _names(types):
    return " or ".join(t.__name__ for t in types)


def _iter_path_entries(value):
    """Yield (genome, intervals) pairs from a data-path list, one level deep or two."""
    if not isinstance(value, list):
        return
    for entry in value:
        if isinstance(entry, list):          # additional_val_data_paths nesting
            for inner in entry:
                yield from _iter_path_entries([inner])
        elif isinstance(entry, dict):
            yield entry.get('genome'), entry.get('intervals')
        elif isinstance(entry, str):
            yield None, entry


def check_paths(config, repo_root, report):
    keys = ['train_data_paths', 'val_data_paths', 'additional_val_data_paths',
            'shap_bg_data_paths', 'shap_fg_data_paths']
    for key in keys:
        if key not in config:
            continue
        for genome, intervals in _iter_path_entries(config[key]):
            for path in (genome, intervals):
                if path is None:
                    continue
                _check_one_path(key, path, repo_root, report)
        for genome, intervals in _iter_path_entries(config[key]):
            if intervals and str(intervals).endswith(('.bed', '.narrowPeak',
                                                      '.narrowPeak.gz', '.bed.gz')):
                if not genome:
                    report.error(f"{key}: '{intervals}' is an interval file but no"
                                 " 'genome:' FASTA was given for it.")

    model_path = config.get('interp_model_path')
    if model_path:
        _check_one_path('interp_model_path', model_path, repo_root, report,
                        missing_is_warning=True)


def _check_one_path(key, path, repo_root, report, missing_is_warning=False):
    path = str(path)
    if path.strip() in ('', 'none', 'None', 'null'):
        return
    if any(marker in path for marker in PLACEHOLDER_MARKERS):
        report.error(f"{key}: unreplaced placeholder path '{path}'.")
        return
    resolved = path if os.path.isabs(path) else os.path.join(repo_root, path)
    if not os.path.exists(resolved):
        msg = f"{key}: file not found '{path}' (looked in {resolved})."
        if missing_is_warning:
            report.warn(msg + " Expected if the model has not been trained yet.")
        else:
            report.error(msg)


def check_lengths(config, report):
    pairs = [('train_data_paths', 'train_targets'),
             ('val_data_paths', 'val_targets'),
             ('additional_val_data_paths', 'additional_val_targets'),
             ('shap_bg_data_paths', 'shap_bg_targets'),
             ('shap_fg_data_paths', 'shap_fg_targets')]
    for paths_key, targets_key in pairs:
        paths, targets = config.get(paths_key), config.get(targets_key)
        if not isinstance(paths, list) or not isinstance(targets, list):
            continue
        if len(paths) != len(targets):
            report.error(f"{paths_key} has {len(paths)} entries but {targets_key}"
                         f" has {len(targets)}; they must match one-to-one, in order.")


def check_layerwise(config, report):
    for count_key, params in (('num_conv_layers', CONV_LAYERWISE),
                              ('num_dense_layers', DENSE_LAYERWISE)):
        count = config.get(count_key)
        if not isinstance(count, int):
            continue
        for param in params:
            value = config.get(param)
            if isinstance(value, list) and len(value) < count:
                report.error(f"'{param}' has {len(value)} values but {count_key}"
                             f" is {count}; it needs at least one value per layer.")


def check_task(config, report):
    targets_are_classes = config.get('targets_are_classes')
    loss = config.get('loss_function')
    class_weight = config.get('class_weight', 'none')

    if targets_are_classes is None:
        report.warn("'targets_are_classes' is not set; it decides regression vs."
                    " classification and should be explicit.")
    elif targets_are_classes is False:
        if loss in CLASSIFICATION_LOSSES:
            report.error(f"targets_are_classes is false but loss_function is '{loss}';"
                         " use a regression loss such as mean_squared_error.")
        if class_weight not in (None, 'none'):
            report.error("class_weight is set on a regression run; the pipeline raises"
                         " unless it is 'none'.")
    else:
        if loss in REGRESSION_LOSSES:
            report.error(f"targets_are_classes is true but loss_function is '{loss}';"
                         " use sparse_categorical_crossentropy.")

    for callback in config.get('early_stopping_callbacks') or []:
        if not isinstance(callback, dict):
            continue
        monitor = callback.get('monitor', '')
        if targets_are_classes is True and 'mean_squared_error' in monitor:
            report.warn(f"early stopping monitors '{monitor}' on a classification run;"
                        " that metric will not be produced.")
        if targets_are_classes is False and ('accuracy' in monitor or 'auc' in monitor.lower()):
            report.warn(f"early stopping monitors '{monitor}' on a regression run;"
                        " that metric will not be produced.")


def check_transformer(config, report):
    use_transformer = config.get('use_transformer')
    num_layers = config.get('num_transformer_layers', 0) or 0
    if use_transformer and num_layers < 1:
        report.error("use_transformer is true but num_transformer_layers is"
                     f" {num_layers}; the model would have no transformer block.")
    if not use_transformer and num_layers >= 1:
        report.warn(f"num_transformer_layers is {num_layers} but use_transformer is"
                    " false; a CNN-only model will be trained.")
    heads = config.get('transformer_num_heads')
    d_model = config.get('transformer_d_model')
    if use_transformer and isinstance(heads, int) and isinstance(d_model, int):
        if heads > 0 and d_model % heads != 0:
            report.error(f"transformer_d_model ({d_model}) is not divisible by"
                         f" transformer_num_heads ({heads}).")


def check_targets(config, report):
    for key in ('train_targets', 'val_targets', 'shap_bg_targets', 'shap_fg_targets'):
        for target in config.get(key) or []:
            if isinstance(target, dict):
                column = target.get('column')
                if column is None:
                    report.error(f"{key}: target dict without a 'column' key: {target}")
                elif isinstance(column, int) and column < 0:
                    report.error(f"{key}: negative column {column}"
                                 " (BED columns are zero-based and non-negative).")


def check_one(path, repo_root):
    report = Report(path)
    config = load(path, report)
    if config:
        check_required(config, report)
        check_lengths(config, report)
        check_layerwise(config, report)
        check_paths(config, repo_root, report)
        check_task(config, report)
        check_transformer(config, report)
        check_targets(config, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('configs', nargs='+', help='config .yaml file(s) to check')
    parser.add_argument('--repo-root', default=os.getcwd(),
                        help='root that relative data paths resolve against'
                             ' (default: current directory)')
    args = parser.parse_args()

    failed = False
    for path in args.configs:
        report = check_one(path, os.path.abspath(args.repo_root))
        print(f"\n{path}")
        for message in report.errors:
            print(f"  ERROR   {message}")
        for message in report.warnings:
            print(f"  WARNING {message}")
        if not report.errors and not report.warnings:
            print("  OK — no problems found.")
        elif not report.errors:
            print(f"  OK with {len(report.warnings)} warning(s).")
        failed = failed or bool(report.errors)

    if failed:
        print("\nFix the errors above before launching a run.")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
