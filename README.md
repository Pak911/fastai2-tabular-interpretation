# fastai2-tabular-interpretation



This is (extended) fastai2-version of my [previous work](https://github.com/Pak911/fastai-shared-notebooks)
This project helps you to interpret tabular models, made with [fastai2](https://github.com/fastai/fastai2)

Some examples of using these methods are made for 2 datasets: well known [Bulldozers dataset](https://www.kaggle.com/c/bluebook-for-bulldozers/data) and transfermarkt's [football players transfer statistics](https://www.kaggle.com/giovannibeli/european-football-transfers-database)
Corresponding interpretations are in [bulldozer](02-interpret.ipynb) and [football](11-football-interpr.ipynb) example notebooks.

Main interpretation methods available are:

- **Dendrogram** -- can help to calculate and visualize features'  correlations which can be used later

[<img src="imgs/01-dendro.png" alt="dendrogramm" width="400"/>](imgs/01-dendro.png)
- **Feature importance** -- can help to calculate relative  and visualize importance of isolated features as well as lists of correlated (connected) features, that were determined earlier

[<img src="imgs/02-FI.png" alt="feature importance" width="400"/>](imgs/02-FI.png)
- **Partial Dependence** -- shows how particular value of a feature influence dependent variable. In what direction we should move this particular feature to minimase or maximize the result

[<img src="imgs/03-PD.png" alt="partial dependence" width="400"/>](imgs/03-PD.png)
- **Waterfall** help to visualize how tabular model came to concluzion in the particular case. How and in what direction each feature value moves the dependent variable

[<img src="imgs/waterfall.png" alt="waterfall chart" width="400"/>](imgs/waterfall.png)
- **Embeddings** -- this chapter helps to visualize embeddings calculated in the model

[<img src="imgs/04-embeddings.png" alt="embeddings" width="400"/>](imgs/04-embeddings.png)

These 5 chapters works nicely with an algorithm based on Jeremy Howard's  [article](https://www.oreilly.com/radar/drivetrain-approach-data-products/). 
In short: 

- We take some task (bulldozer's sales), make it's model (fastai [tabular model creation](01-train.ipynb)). 
- Then we determine what features (***feature importance***) influence our value the most (let's say we want sell our bulldozer as high as possible). 
- Optionally dividing some features into groups (***dendrogram***). 
- Then we look at our task and find the features we can change in the real word from the top-important features (for example we can change in what state we sell our bulldozer or some other features, in fact I know nothing about bulldozers market in US :( )
- After that we find the most useful for us value of this feature. In whole dataset (***partial dependence***) or in our particular case (***waterfall***). The last one also help us to determine what values drive price up or down the most.
- Having this information and knowing what we can really change, we can optimize our bulldozer's sell price

This work is based on [my previous notebook](https://github.com/Pak911/fastai-shared-notebooks) which in turn was based on [Jeremy Howard's lectures](https://www.youtube.com/watch?v=YSFG_W8JxBo). 
Also some parts of this work are inspired by [Zachary Mueller's lectures](https://www.youtube.com/playlist?list=PLFDkaGxp5BXDvj3oHoKDgEcH73Aze-eET) especially [tabular interpretation lesson](https://www.youtube.com/watch?v=XoWX_YOrtPg&list=PLFDkaGxp5BXDvj3oHoKDgEcH73Aze-eE)

Restrictions:
I've tested it for regression-based models only. Don't think it will work for classification without some refactoring



