import numpy as np
import keras

def lr_schedule(epoch, lr_init=0.01, schedule=((50,0.1), (100, 0.01), (150, 0.001))):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs
        lr_init (int): Initial learning rate
        schedule (list of int): epoch and factor, first and second arguments. Factor will be update the initial learning rate (lr = lr*factor)

    # Returns
        lr (float32): learning rate
    """
    lr = lr_init
    for i in range(0, len(schedule)):
        ep = schedule[i][0]
        if epoch == ep:
            lr = lr*schedule[i][1]
            print('Learning rate: ', lr)
            return lr

    print('Learning rate: ', lr)
    return lr

def custom_stopping(value=0.5, verbose=0):
    early = keras.callbacks.EarlyStoppingByLossVal(monitor='val_loss', value=value, verbose=verbose)
    return early

class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='val_acc', value=0.95, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        # if current is None:
        # warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

