from flcore.trainmodel.models import LSTM
from flcore.trainmodel.rul_models import LSTM_v2_RUL, MLP_LSTM_MLP, AFTConv2D, AttBiGRU, RNN_RUL, Chen_CNN_RUL

class RULModelFactory:
    @staticmethod
    def create_model(args):
        if args.model == "LSTM_RUL":
            # configuration used in "A Federated Learning-Based Industrial Health Prognostics 
            # for Heterogeneous Edge Devices Using Matched Feature Extraction"
            args.local_learning_rate = 0.01
            args.batch_size = 16
            return LSTM(
                input_size=args.input_size,
                hidden_sizes=[256],
                output_size=1,
                dropout_prob=getattr(args, 'dropout_prob', 0.0),
                output_clip_0_1=args.output_clip_0_1, 
                output_sigmoid=args.output_sigmoid,
            )
        
        if args.model == "LSTM_v2_RUL":
            # configuration used in "Using Federated Machine Learning in Predictive Maintenance of Jet Engines"
            # batch size: 32; learning rate: 0.001; optimizer: AdamW
            args.local_learning_rate = 0.01
            args.batch_size = 16
            return LSTM_v2_RUL(
                input_size=args.input_size,
                lstm_layers=4,
                dense_layers=4,
                units=64,
                layer_dropout=0.1,
                recurrent_dropout=0.2,
                gaussian_noise=0.01,
                output_clip_0_1=args.output_clip_0_1, 
                output_sigmoid=args.output_sigmoid,
            )
        
        elif args.model == "MLP_LSTM_MLP_RUL":
            # batch size: 14; learning rate: 0.0002; weight decay: 0.001; optimizer: AdamW
            # 200 global rounds; 1 local epoch
            # trained for 300 epochs; best model selected by lowest validation loss
            #args.local_learning_rate = 2e-4
            args.local_learning_rate = 0.01
            args.batch_size = 64
            return MLP_LSTM_MLP(
                input_size=args.input_size,
                feature_dim_out=128,
                lstm_hidden=128,
                lstm_layers=1,
                head_hidden=[128, 64],
                return_scalar=False,
                output_clip_0_1=args.output_clip_0_1, 
                output_sigmoid=args.output_sigmoid,
            )
        
        elif args.model == "AFT_RUL":
            # learning rate: 0.004; optimizer: Adam
            # 20 global rounds; 18 local epochs
            #args.local_learning_rate = 4e-3
            args.local_learning_rate = 0.01
            args.batch_size = 32
            return AFTConv2D(
                input_size=args.input_size,
                window_size=args.window_size, # required for the learnable positional bias, which is a [window_size, window_size] matrix
                output_clip_0_1=args.output_clip_0_1, 
                output_sigmoid=args.output_sigmoid,
            )
        
        elif args.model == "AttBiGRU_RUL":
            # batch size: 64; learning rate: 0.001
            # 50 global rounds; 3 local epochs; 10 window size
            #args.local_learning_rate = 1e-3
            args.local_learning_rate = 0.01
            args.batch_size = 32
            return AttBiGRU(
                input_size=args.input_size,
                ws=args.window_size,
                gru_hidden=32,
                attn_heads=4,
                output_clip_0_1=args.output_clip_0_1, 
                output_sigmoid=args.output_sigmoid,
            )
        
        elif args.model == "RNN_RUL":
            # batch size: 64; learning rate: 0.0005; optimizer: Adam  
            # 10 global rounds; 100 local epochs; 5 nodes; 100 window size; #features: 16
            # convergenza entro 300 epoche
            #args.local_learning_rate = 5e-4
            args.local_learning_rate = 0.001
            args.batch_size = 16
            return RNN_RUL(
                input_size=args.input_size,
                fc_hidden=40,
                output_clip_0_1=args.output_clip_0_1, 
                output_sigmoid=args.output_sigmoid,
            )
        
        elif args.model == "Chen_CNN_RUL":
            # optimizer: AdaMod
            # window size: 30 (FD001) and 15 (FD004); #features: 14
            # retraining rounds 80 (FD001) and 100 (FD004)
            args.local_learning_rate = 0.01
            args.batch_size = 16
            return Chen_CNN_RUL(
                input_size=args.input_size,
                window_size=args.window_size,
                conv_channels=[16, 16, 16, 16, 16, 16], # not defined in the paper
                fc_dropout=getattr(args, "dropout_prob", 0.0),
                output_clip_0_1=args.output_clip_0_1, 
                output_sigmoid=args.output_sigmoid,
            )

        else:
            raise NotImplementedError(f"Model {args.model} is not supported by RULModelFactory.")
