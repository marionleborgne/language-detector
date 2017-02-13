# Language detector exercise

Hosted on [Heroku](https://language-detector.herokuapp.com/).

Uses [custom buildbpack](https://github.com/thenovices/heroku-buildpack-scipy) for scipy dependency.

This is a simple Flask app that will predict the language of a sentence using sci-kit Learn.


## How to train a model once the model is finished.

Create a new virtual environment:
```
virtualenv venv
source venv/bin/activate
```

Add heroku remote:
```
heroku https://git.heroku.com/nameless-fortress-57367.git
```

Run:

```
cd ml/
python language_detector.py
```

Deploy to heroku:

```
git add .
git commit -m 'yo'
git push heroku master
```

Log into heroku and launch the app.
