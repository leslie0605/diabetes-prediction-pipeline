
import os
import argparse

from trainer import model  # Assume model.py is under trainer/ directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Vertex AI sets the model artifact path in AIP_MODEL_DIR,
    # but you can also provide a custom default or override it.
    parser.add_argument('--model-dir', dest='model-dir',
                        default=os.environ.get('AIP_MODEL_DIR', 'exported_model'),
                        type=str,
                        help='Path to export the trained model.')

    parser.add_argument('--data-file', dest='data-file',
                        default='data/diabetes_prediction_dataset.csv',
                        type=str,
                        help='Path to the CSV dataset file.')

    parser.add_argument('--batch-size', dest='batch-size',
                        default=32, type=int,
                        help='Batch size for training.')

    parser.add_argument('--epochs', dest='epochs',
                        default=5, type=int,
                        help='Number of training epochs.')

    parser.add_argument('--dropout', dest='dropout',
                        default=0.5, type=float,
                        help='Dropout rate.')

    args = parser.parse_args()
    hparams = args.__dict__

    model.train_evaluate(hparams)
