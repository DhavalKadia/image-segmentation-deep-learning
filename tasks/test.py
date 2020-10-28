from package.tasks.TestClass import TestClass

def main():
    # object of the Test Class
    object = TestClass(test_model_name='pretrained-model.h5', test_whole=True)
    # start evaluation
    object.test()

if __name__ == "__main__":
    main()
