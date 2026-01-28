from flcore.trainmodel.models import LSTM

class RULModelFactory:
    @staticmethod
    def create_model(args):
        if args.model == "LSTM":
            return LSTM(
                input_size=getattr(args, 'input_size', 14),
                hidden_sizes=getattr(args, 'hidden_sizes', [128, 64]),
                output_size=1,
                dropout_prob=getattr(args, 'dropout_prob', 0.0)
            )
        else:
            raise NotImplementedError(f"Model {args.model} is not supported by RULModelFactory.")
