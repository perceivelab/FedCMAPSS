from flcore.trainmodel.models import LSTM
from flcore.trainmodel.rul_models import AFTConv2D, Chen_CNN_RUL, AttBiGRU

class RULModelFactory:
    @staticmethod
    def create_model(args):
        if args.model == "LSTM_RUL":
            # configuration used in "A Federated Learning-Based Industrial Health Prognostics 
            # for Heterogeneous Edge Devices Using Matched Feature Extraction"
            return LSTM(
                input_size=args.input_size,
                hidden_sizes=[256],
                output_size=1,
                dropout_prob=getattr(args, 'dropout_prob', 0.0)
            )
        elif args.model == "AFT_RUL":
            return AFTConv2D(
                input_size=args.input_size,
                window_size=args.window_size, # required for the learnable positional bias, which is a [window_size, window_size] matrix 
            )
        elif args.model == "Chen_CNN_RUL":
            return Chen_CNN_RUL(
                input_size=args.input_size,
                window_size=args.window_size,
                conv_channels=[16, 16, 16, 16, 16, 16], # not defined in the paper
                fc_dropout=getattr(args, "dropout_prob", 0.0),
            )
        elif args.model == "AttBiGRU_RUL":
            return AttBiGRU(
                input_size=args.input_size,
                ws=args.window_size,
                gru_hidden=32, 
                attn_heads=4
            )
        else:
            raise NotImplementedError(f"Model {args.model} is not supported by RULModelFactory.")
