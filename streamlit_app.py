# import pandas as pd
# import numpy as np
import pickle
import streamlit as st
# from PIL import Image
# import streamlit.components.v1 as components

od = pickle.load(open('labelencoder.pkl', 'rb'))
model = pickle.load(open('mlr.pkl', 'rb'))

def mlr_prediction(engine, hp, vol, sp, wt):  
    #a = str(engine)
    b = float(hp)
    c = float(vol)
    d = float(sp)
    e = float(wt)
    
    prediction = model.predict([[od.transform([engine]), b, c, d, e]])
    print(prediction)
    return abs(prediction)

def main():
      # giving the webpage a title
    st.title("Car fuel Efficiency")
      
    # We define the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
     
    """
      
    # This line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # The following code creates text boxes in which the user can enter 
    # the data required to make the predictions
    engine = st.text_input("Engine Type (Please mention Engine Type: petrol, hybrid, diesel, lpg, or cng)")
    hp = st.number_input("HP(Range 49-322)")
    vol = st.number_input("VOL(Range 50-160)")
    sp = st.number_input("SP(Range 16-52)")
    wt = st.number_input("WT(Range 16-52)")
    result =""
      
    # The code below ensures that when the 'Predict' button is clicked, 
    # the mlr_prediction function defined above is called.
    if st.button("Predict"):
        result = mlr_prediction(engine, hp, vol, sp, wt )
    st.success('The output is : {}'.format(result[0]))
     
if __name__=='__main__':
    main()