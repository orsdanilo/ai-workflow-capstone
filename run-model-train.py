import os
from model import model_train, model_load

def main():
    
    ## train the model
    print("TRAINING MODELS")
    data_dir = os.path.join(".","data","cs-train")
    model_train(data_dir,test=True)

    ## load the model
    print("LOADING MODELS")
    all_data, all_models = model_load()
    print("... models loaded: ",",".join(all_models.keys()))
    
    print("model training complete.")


if __name__ == "__main__":

    main()
