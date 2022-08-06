from networks import *
from train_nn import train_nn, train_dqn
from datasets import *
from DQN import DQNAgent
from tqdm import tqdm
from utils import unsupervised_learning_test

if __name__ == '__main__':
    # Step 1: determine the device on which the model is trained and some parameters:
    # commands:
    OVERWRITE_MODEL = True
    OVERWRITE_DATASET = True
    # parameters:
    num_MUs = 10
    num_BDs = 10
    num_batches = 100  # 1000_000
    train_batch_size = 500
    test_batch_size = 500
    max_epochs = 100
    # file paths:
    model_name = 'DNN'
    model_path = './model/dnn_solution'
    train_data_path = './data/train_data.tar'
    test_data_path = './data/test_data.tar'
    result_save_path = './results'

    # Using device:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Step 2: get the train and the test datasets.
    G, h, f = generate_dataset(num_BDs, num_MUs, num_channel_realizations=num_batches, region=(100, 100))
    train_set = ChannelInfoDatasetTrain(G, f)
    val_set = ChannelInfoDatasetTest(G, f)

    # Step 3: create the NN model to be trained and train the model.
    # net = UserAssociationNet(input_dims=num_BDs, output_dims=num_BDs).to(device)
    # net = train_nn(net, train_set=train_set, val_set=val_set, device=device,
    #                batch_size=train_batch_size, max_epochs=max_epochs) if OVERWRITE_MODEL \
    #     else load_model(model_path, model_name)

    # Step 4: test the model
    # test_loader = data.DataLoader(val_set, batch_size=500, pin_memory=True, shuffle=False, drop_last=True)

    # Step 5: save & plot
    # unsupervised_learning_test(net, test_set=val_set,
    #                            path=result_save_path+'unsupervised_learning', save_res=True, plot=True)

    # Step 6: train DQN
    DQN_Agent = agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.001,
                                 state_dims=num_BDs+1, num_actions=num_BDs, action_dims=1)
    train_dqn(DQN_Agent, train_set)

