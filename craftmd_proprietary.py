import os
import pandas as pd
from multiprocessing import Pool

proprietary_model_name = "<Insert Model Name>"
proprietary_model_endpoint = "<Insert Model API Endpoint>"
proprietary_model_key = "<Insert Model API Bearer Token>"


from src.utils import get_choices
from src.craftmd import craftmd_proprietary
from src.configurations import EvalConfig, ModelAPIConfig



if __name__ == "__main__":
    
    dataset = pd.read_csv("./data/usmle_and_derm_dataset.csv", index_col=0)

    dataset = dataset.head(100)

    # Initialize with specific configurations
    model_api_config = ModelAPIConfig(
        name=proprietary_model_name,
        endpoint=proprietary_model_endpoint,
        key=proprietary_model_key
    )

    eval_config = EvalConfig(
        answer_type_config={
            "MCQ": False,
            "FRQ": True
        },
        conversation_type_config={
            "vignette": False,
            "multi_turn": True,
            "single_turn": False,
            "summarized": False
        }
    )

    # Set number of threads for parallelization
    num_cpu = 7
    
    cases = [(dataset.loc[idx,"case_id"], 
              dataset.loc[idx,"case_vignette"], 
              dataset.loc[idx,"category"],
              get_choices(dataset,idx)) for idx in dataset.index]

    path_dir = f"./results/{model_api_config.name}"
    os.makedirs(path_dir, exist_ok=True)
        
    with Pool(num_cpu) as p:
        p.starmap(craftmd_proprietary, [(x, path_dir, model_api_config, eval_config) for x in cases])