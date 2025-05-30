from CNN.load import main as cnn_main
from RNN.test import main as rnn_main

if __name__ == "__main__":
    print("Running best models for CNN...")
    cnn_main()
    
    print("Running best models for RNN...")
    rnn_main()