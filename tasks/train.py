from package.tasks.TrainClass import TrainClass

def main():
    # object of the Train Class
    object = TrainClass(learning_rate=1e-4, train_batch_size=8, training_steps=5, epochs=10, utilize_whole=True,
                    transfer_learning=True, transfer_model_name='pretrained-model.h5', next_model_title='model_2')
    # start training
    object.train()

if __name__ == "__main__":
    main()
