from flask import Flask , render_template , request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

df=pd.read_csv('static/movie_finalized_data.csv')
cv =CountVectorizer()
vec_matrix =cv.fit_transform(df['combined'])
similar =cosine_similarity(vec_matrix)

def recommend_movies(movie):
    if movie not in df['movie_title'].unique():
        return []
    else:
        i=df.loc[df['movie_title'] == movie ].index[0]
        lst =list(enumerate(similar[i]))
        lst =sorted(lst , key =lambda x:x[1] ,reverse =True)
        lst=lst[1:11]
        result=[]
        for i in range(len(lst)):
            a=lst[i][0]
            result.append(df['movie_title'][a])
        return result


app=Flask(__name__)
PEOPLE_FOLDER = os.path.join('static')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


@app.route('/')
@app.route('/home')
def home():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'i.jpg')
    return render_template('index.html', user_image =full_filename)
@app.route('/predict',methods=["POST"])
def predict():
    mov_name= request.form.get('movie')
    pred =recommend_movies(movie=mov_name)
    return render_template('index.html', prediction_text ="Recommend{}".format(pred), data=pred, len=len(pred))


if __name__ == '__main__':
    app.run(debug=True , port='8000')