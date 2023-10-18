# Boston House Pricing Prediction (MLOps)

### Software and tools requirements

1. [Github Account](https://github.com)
2. [Heroku Account](https://heroku.com)
3. [VS Code IDE](https://code.visualstudio.com)
4. [Git CLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)

Create a bew environment
```
conda create -p venv python==3.7 -y
```

## What have I done for this learning project?

Developed an ML model from a simple dataset such as Boston House Pricing and deployed the model in a Heroku server through a Docker container.

- First the data is analysed through EDA in a Jupyter Notebook. Also, a linear regression model is created and exported as a Pickle file.

- Then an application is developed locally. An app.py script uses Flask library to make a web application through a home.html file (basic front-end app).

- After checking it works, it is deployed in a Heroku server. A Procfile file is created, indicating the name of the app and gunicorn to be able to run Python on Heroku.

- Finally, once it is deployed to the Heroku server, the application is converted into a Docker image to be run in a Docker container from the same Heroku server.

## BONUS: Lessons learnt

- It is very easy to make mistakes when writing the commands on the Dockerfile, main.yaml and Procfile.

- This is my first ML app deployment and I think it is a must to know about this as a data scientist.