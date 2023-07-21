import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Alzheimer Stage Prediction", layout='wide')

st.markdown("<h1 style='text-align: center; color: gray;'>Stage of Alzheimer </h1>", unsafe_allow_html=True)

page_style = '''
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
            '''

st.markdown(page_style,unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

model = tf.keras.models.load_model("Trained_Model2.h5")

selected_tab = option_menu(
    menu_title = None,
    options = ['About Alzheimers', 'Statistics',"How to Use", 'Alzheimers\'s Detection',"Get in Touch with Me"],
    icons = ['house-door','graph-up','file-earmark-text','binoculars','envelope-open'],
    menu_icon = 'cast',
    default_index = 0,
    orientation = 'horizontal',
    styles={
        "icon": {"font-size": "20px"},
        "nav-link": {"font-size": "18px"}
    }
)



if selected_tab == 'Alzheimers\'s Detection':

    col1,col2,col3,col4,col5 = st.columns(5)
    with col2:
        st.markdown("<h5 style='text-align: center; color: gray;'>Normal Brain</h5>", unsafe_allow_html=True)
        st.image("https://upload.wikimedia.org/wikipedia/commons/b/b2/MRI_of_Human_Brain.jpg")

    st.write("##")
    data = st.file_uploader("Upload an MRI Image of the brain",type = ['png','jpeg','jpg'])
    if data:
        with col4:
            st.markdown("<h5 style='text-align: center; color: gray;'>Uploaded Image</h5>", unsafe_allow_html=True)
            st.image(tf.keras.utils.load_img(data, target_size = (1433,1534)))
    c1,c2,c3,c4,c5 = st.columns(5)
    with c3:
        butt = st.button("Click to check")
    if butt:
        if data:
            img = tf.keras.utils.load_img(data, target_size = (128,128),color_mode='rgb')
            img = tf.convert_to_tensor(img)
            pred = np.argmax(tf.nn.softmax(model.predict(tf.expand_dims(img, 0))[0]))
            with open("classes.pkl","rb") as f:
                d = pickle.load(f)
            st.info("The model predicts the image as "+d[pred])



elif selected_tab == 'About Alzheimers':
    st.markdown("<h6 style='text-align: center; color: gray;'>Alzheimer's disease (AD) is a neurodegenerative disease that usually starts slowly and progressively worsens, and is the cause of 60–70% of cases of dementia.\
         The most common early symptom is difficulty in remembering recent events.\
         As the disease advances, symptoms can include problems with language, disorientation (including easily getting lost), mood swings, loss of motivation, self-neglect, and behavioural issues.\
         As a person's condition declines, they often withdraw from family and society. Gradually, bodily functions are lost, ultimately leading to death.\
         Although the speed of progression can vary, the typical life expectancy following diagnosis is three to nine years.<h6>", unsafe_allow_html=True)
    
    c1,c2= st.columns([2,2])
    with c1:
        st.write("##")
        st.image("https://www.jax.org/-/media/AEC74EFDF0234A03AA44A4D461BC5E81.jpg",width = 500)
    with c2:
        # st.markdown(br,unsafe_allow_html=True)
        st.write("##")
        st.markdown("<h6 style='text-align: center; color: gray;'>The cause of Alzheimer's disease is poorly understood.\
                 There are many environmental and genetic risk factors associated with its development. \
                 The strongest genetic risk factor is from an allele of APOE. Other risk factors include a history of head injury, clinical depression, and high blood pressure.\
                 The disease process is largely associated with amyloid plaques, neurofibrillary tangles, and loss of neuronal connections in the brain.\
                 A probable diagnosis is based on the history of the illness and cognitive testing, with medical imaging and blood tests to rule out other possible causes.\
                 Initial symptoms are often mistaken for normal brain aging.\
                 Examination of brain tissue is needed for a definite diagnosis, but this can only take place after death.\
                 Good nutrition, physical activity, and engaging socially are known to be of benefit generally in aging, and may help in reducing the risk of cognitive decline and Alzheimer's.\
                <h6>", unsafe_allow_html=True)
    c1,c2 = st.columns([2,2])
    with c1:
        st.image("https://www.mycirclecare.com/wp-content/uploads/2018/06/Effect-of-Alzheimer-by-Stages.jpg", width = 500)

    with c2:
        st.markdown("<h6 style='text-align: center; color: gray;'>No treatments can stop or reverse its progression, though some may temporarily improve symptoms. Affected people become increasingly reliant on others for assistance, often placing a burden on caregivers.\
            The pressures can include social, psychological, physical, and economic elements. Exercise programs may be beneficial with respect to activities of daily living and can potentially improve outcomes.\
            Behavioural problems or psychosis due to dementia are often treated with antipsychotics, but this is not usually recommended, as there is little benefit and an increased risk of early death.\
            As of 2020, there were approximately 50 million people worldwide with Alzheimer's disease. It most often begins in people over 65 years of age, although up to 10% of cases are early-onset impacting those in their 30s to mid-60s. It affects about 6% of people 65 years and older, and women more often than men.\
            The disease is named after German psychiatrist and pathologist Alois Alzheimer, who first described it in 1906. Alzheimer's financial burden on society is large, with an estimated global annual cost of US$1 trillion. It is ranked as the seventh leading cause of death in the United States.\
            <h6>", unsafe_allow_html=True)
    
    st.markdown("<h6 style='text-align: center; color: white;'>This information was made available using Google Search and Wikipedia</h6>", unsafe_allow_html=True)


elif selected_tab == 'Statistics':
    c1, c2 = st.columns(2)
    with c1:
        st.image("imgs/5.png")
    with c2:
        st.image("imgs/12.png",width = 600)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("##")
        st.image("imgs/1.png")
    with c2:
        st.image("imgs/3.png")
    with c3:
        st.write("##")
        st.image("imgs/11.png")
    c1, c2 = st.columns(2)
    with c1:
        st.image("imgs/4.png")
        st.image("imgs/9.png")
    with c2:
        st.image("imgs/13.png")
        st.write("##")
        st.image("imgs/10.png")
        st.write("##")
    c1,c2,c3 = st.columns([1,8,1])
    with c2:
        st.image("imgs/2.png", width = 1000)

    st.markdown("<h6 style='text-align: center; color: white;'>All this Statistics are taken from the special report \"2023 ALZHEIMER’S DISEASE FACTS AND FIGURES\" by Alzheimer's Association</h6>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.write("https://www.alz.org/media/documents/alzheimers-facts-and-figures.pdf")


elif selected_tab == 'How to Use':
    st.write("My Model is trained on more then 2.6 Million Parameters. It has a weighted average F1 Score of 0.97.")
    st.write("My model divides the stages of Alzeimer in 4 different stages (according to their severity):")
    st.write("1. Non Demented")
    st.write("2. Very Mild Demented")
    st.write("3. Mild Demented")
    st.write("4. Moderate Demented")
    st.write("##")
    st.write("To use this Machine Learning model:")
    st.write("1. You first need a soft copy of MRI of your brain.  {In case you don't have any image https://shorturl.at/ixyHO }")
    st.write("2. Upload the image on the Next tab known as Alzheimer's Detection")
    st.write("3. The model will out the prediction below the button in a Blue coloured dialog Box")



elif selected_tab == 'Get in Touch with Me':
    st.write("##")
    col1, col2, col3 = st.columns([1,1,2])
    # st.subheader(":mailbox: Get in Touch With Me...!")
    contact_form = '''
        <form action="https://formsubmit.co/tokas.2sonu@gmail.com" method="POST">
            <input type="hidden" name="_autoresponse" value="Thank You for spending your valuable time on my website. I will contact you soon.">
            <input type="hidden" name="_template" value="table">
            <input type="hidden" name="_next" value="https://alzheimerdiseasedetection.streamlit.app/">
            <input type="text" name="name" id = 'input' placeholder = "Your Name" required>
            <input type="email" name="email" id = 'input' placeholder = "Your Email" required>
            <textarea name = 'message' id = 'input' placeholder = 'Your Message' required></textarea>
            <button onclick="document.getElementById('input').value = ''" type="submit">Send</button>
        </form>
    '''

    with col1:
        st.subheader("Meet the Developer")
        pp = Image.open("profile pic/profile-pic.png")
        st.image(pp, output_format='PNG')
        st.write("  Developer : Vivek Goel")


    with col2:
        for _ in range(6):
            st.write("")

        st.write("Connect with me at:")
        st.write("[Github](https://github.com/vivek-2567)")
        st.write("[Linkedin](https://www.linkedin.com/in/vivek-goel-0207/)")
        st.write("Mail me @")
        st.write("[Mail](mailto:vivekgoel0207@gmail.com)")

    with col3:
        st.subheader("Send me a Message :rocket:")
        st.markdown(contact_form,unsafe_allow_html=True)
        local_css("style/style.css")
