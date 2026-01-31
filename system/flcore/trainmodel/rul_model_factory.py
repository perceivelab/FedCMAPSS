from flcore.trainmodel.models import LSTM
from flcore.trainmodel.rul_models import LSTM_v2_RUL, AFTConv2D, AttBiGRU, RNN_RUL, Chen_CNN_RUL

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
                dropout_prob=getattr(args, 'dropout_prob', 0.0),
            )
        
        if args.model == "LSTM_v2_RUL":
            # configuration used in "Using Federated Machine Learning in Predictive Maintenance of Jet Engines"
            return LSTM_v2_RUL(
                input_size=args.input_size,
                lstm_layers=4,
                dense_layers=4,
                units=64,
                layer_dropout=0.1,
                recurrent_dropout=0.2,
                gaussian_noise=0.01,
            )
        
        elif args.model == "AFT_RUL":
            return AFTConv2D(
                input_size=args.input_size,
                window_size=args.window_size, # required for the learnable positional bias, which is a [window_size, window_size] matrix 
            )
        
        elif args.model == "AttBiGRU_RUL":
            return AttBiGRU(
                input_size=args.input_size,
                ws=args.window_size,
                gru_hidden=32,
                attn_heads=4,
            )
        
        elif args.model == "RNN_RUL":
            return RNN_RUL(
                input_size=args.input_size,
                fc_hidden=40,
            )
        
        elif args.model == "Chen_CNN_RUL":
            return Chen_CNN_RUL(
                input_size=args.input_size,
                window_size=args.window_size,
                conv_channels=[16, 16, 16, 16, 16, 16], # not defined in the paper
                fc_dropout=getattr(args, "dropout_prob", 0.0),
            )

        else:
            raise NotImplementedError(f"Model {args.model} is not supported by RULModelFactory.")
