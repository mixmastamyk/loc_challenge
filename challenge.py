from flask import Flask
from flask import redirect, request, url_for
from flask import render_template as render


#~ from .patentSimilarityApp import get_input, full_pipeline
from .patentSimilarityApp_sim import get_input, full_pipeline


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def start_page():

    if request.method == 'POST':

        return redirect(url_for('results_page'))
    else:
        return render('start.html')


@app.route('/results/')
def results_page(name=None):

    results = full_pipeline(get_input('foo bar baz '))

    return render('results.html', results=results)
