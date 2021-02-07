# Epidemic spreading Complex Networks 2020
This repository contains code for Prediction of the effects of epidemic spreading with graph neural networks paper
from the Complex Networks 2020 conference. The paper can be found [here](https://link.springer.com/chapter/10.1007/978-3-030-65347-7_35?fbclid=IwAR0Ng3BGF014jQOe_1sWvz858dELltSV7zmCpoRdnw3HVZFfL03onmqaE2g)
and cited as:

```
@InProceedings{meznar2020spreading,
    author="Me{\v{z}}nar, Sebastian and Lavra{\v{c}}, Nada and {\v{S}}krlj, Bla{\v{z}}",
    editor="Benito, Rosa M. and Cherifi, Chantal and Cherifi, Hocine
        and Moro, Esteban and Rocha, Luis Mateus and Sales-Pardo, Marta",
    title="Prediction of the Effects of Epidemic Spreading with Graph Neural Networks",
    booktitle="Complex Networks {\&} Their Applications IX",
    year="2021",
    publisher="Springer International Publishing",
    address="Cham",
    pages="420--431",
    isbn="978-3-030-65347-7"
}
```

An overview of the approach can be seen in the image below.

![algorithm overview](https://github.com/smeznar/Epidemic-spreading-CN2020/blob/master/images/overview.png)

# Running the code

Intstall the required python packages using the command:

```
pip install -r requirements.txt
```

The evaluation of machine learning methods can be ran using the command:

```
python evaluate_epidemics.py  
```

Simulations from the original paper are marked with name old. The results of these simulations is approximately 
shown in the two tables below.

![results max node](https://github.com/smeznar/Epidemic-spreading-CN2020/blob/master/images/results_max.png)

![results time](https://github.com/smeznar/Epidemic-spreading-CN2020/blob/master/images/results_time.png)

# Interpreting CABoost model

The prediction can be interpreted using tools such as SHAP. An example of this can be tested by running the script 
interpreting_effects.py as:

```
python interpreting_effects.py
```

Running the code gives a waterfall plot such as the one on the image below

![shap](https://github.com/smeznar/Epidemic-spreading-CN2020/blob/master/images/shap.png)

# Simulations

Additional simulation data can be created using the run create_data.py script. The format of simulation data "{time} {score}"
but should be change to "{node} {time} {score}" for usage.
