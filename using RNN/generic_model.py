import torch
import torch.nn as nn
import os


class generic_model(nn.Module):
    """
    contains basic functions for storing and loading a model
    """

    def __init__(self, config):

        super(generic_model, self).__init__()

        self.config_file = config

    def loss(self, predicted, truth):

        return self.loss_func(predicted, truth)

    # save model, along with loss details and testing accuracy
    # best is the model which has the lowest test loss. This model is used during feature extraction
    def save_model(self, is_best, epoch, train_loss, test_loss, rnn_name, layers, hidden_dim):

        base_path = self.config_file['models']
        if is_best:
            filename = base_path + 'best_' + '_'.join([rnn_name, str(layers), str(hidden_dim)]) + '.pth'
        else:
            filename = base_path + str(epoch) + '_' + '_'.join([rnn_name, str(layers), str(hidden_dim)]) + '.pth'

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, filename)

        print("Saved model")

    # Loads saved model for resuming training or inference
    def load_model(self, mode, rnn_name, layers, hidden_dim, epoch=None):

        # if epoch is given, load that particular model, else load the model with name 'best'
        if mode == 'test' or mode == 'test_one':

            try:
                if epoch is None:
                    filename = self.config_file['models'] + 'best_' + '_'.join(
                        [rnn_name, str(layers), str(hidden_dim)]) + '.pth'
                else:
                    filename = self.config_file['models'] + str(epoch) + '_'.join(
                        [rnn_name, str(layers), str(hidden_dim)]) + '.pth'

                checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
                # load model parameters
                print("Loading:", filename)
                self.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Loaded pretrained model from:", filename)

            except:
                print("Couldn't find model for testing")
                exit(0)
        # train
        else:
            # if epoch is given, load that particular model else, load the model trained on the most number of epochs
            # e.g. if dir has 400, 500, 600, it will load 600.pth
            if epoch is not None:
                filename = self.config_file['models'] + str(epoch) + '_' + '_'.join(
                    [rnn_name, str(layers), str(hidden_dim)]) + '.pth'
            else:
                directory = [x.split('_') for x in os.listdir(self.config_file['models'])]
                to_check = []
                for poss in directory:
                    try:
                        to_check.append(int(poss[0]))
                    except:
                        continue

                if len(to_check) == 0:
                    print("No pretrained model found")
                    return 0, [], []
                # model trained on the most epochs
                filename = self.config_file['models'] + str(max(to_check)) + '_' + '_'.join(
                    [rnn_name, str(layers), str(hidden_dim)]) + '.pth'

            # load model parameters and return training/testing loss and testing accuracy
            checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print("Loaded pretrained model from:", filename)

            return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['test_loss']
