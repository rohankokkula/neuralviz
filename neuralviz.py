import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from neuralogic import NeuralNetwork,NeuronLayer, Neuron
import pandas as pd
import time
st.set_page_config(
    page_title="NeuralViz.",
    layout="centered",
    initial_sidebar_state="expanded")
st.sidebar.image("logo.png",use_column_width=True)

def draw_neural_net(layer_sizes,wh,opac,wts):

    val=np.random.choice(range(10), 1, replace=False)
    fig = plt.figure(num=val)
    fig.set_size_inches(wh, wh)
    ax, left, right, bottom, top=fig.gca(), .1, .9, .1, .9
    ax.axis("off")
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Input-Arrows
    layer_top_0 = v_spacing*(layer_sizes[0] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[0]):
        plt.arrow(left-0.18, layer_top_0 - m*v_spacing, 0.12, 0,  lw =1, head_width=0.01, head_length=0.02)
    
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/8.,
                                color='#2b2b2b', zorder=4)
            if n == 0:
                plt.text(left-0.14, layer_top - m*v_spacing-0.005, r'$X_{'+str(m+1)+'}$', fontsize=15)
            elif (n_layers == 3) & (n == 1):
                plt.text(n*h_spacing + left+0.00, layer_top - m*v_spacing+ (v_spacing/8.+0.01*v_spacing), r'$H_{'+str(m+1)+'}$', fontsize=15)
            elif n == n_layers -1:
                plt.text(n*h_spacing + left+0.11, layer_top - m*v_spacing, r'$y_{'+str(m+1)+'}$', fontsize=15)
            ax.add_artist(circle)
    # Bias-Nodes
    for n, layer_size in enumerate(layer_sizes):
        if n < n_layers -1:
            x_bias = (n+0.5)*h_spacing + left
            y_bias = top + 0.005
            circle = plt.Circle((x_bias, y_bias), v_spacing/8., color='#f63366', zorder=4)
            if(wts):
                plt.text(x_bias-(v_spacing/8.+0.10*v_spacing+0.01)-0.07, y_bias,f"B{n+1}=0.40", fontsize=15)
            ax.add_artist(circle)   
    # Edges
    # Edges between nodes
    
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k',alpha=opac)
                ax.add_artist(line)
                xm = (n*h_spacing + left)
                xo = ((n + 1)*h_spacing + left)
                ym = (layer_top_a - m*v_spacing)
                yo = (layer_top_b - o*v_spacing)
                rot_mo_rad = np.arctan((yo-ym)/(xo-xm))
                rot_mo_deg = rot_mo_rad*180./np.pi
                xm1 = xm + (v_spacing/8.+0.05)*np.cos(rot_mo_rad)
                if n == 0:
                    if yo > ym:
                        ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing/8.+0.07)*np.sin(rot_mo_rad)
                else:
                    if yo > ym:
                        ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing/8.+0.04)*np.sin(rot_mo_rad)
                if(n==0):
                    val="ih"
                else:
                    val="ho"        
                if(wts):
                    plt.text( xm1, ym1,f"w{val}_{m+1}{o+1}",rotation = rot_mo_deg, fontsize = 12)
                
    # Edges between bias and nodes
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if n < n_layers-1:
            layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
            layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        x_bias = (n+0.5)*h_spacing + left
        y_bias = top + 0.005 
        for o in range(layer_size_b):
            line = plt.Line2D([x_bias, (n + 1)*h_spacing + left],
                          [y_bias, layer_top_b - o*v_spacing], c='k',alpha=opac)
            ax.add_artist(line)
            xo = ((n + 1)*h_spacing + left)
            yo = (layer_top_b - o*v_spacing)
            rot_bo_rad = np.arctan((yo-y_bias)/(xo-x_bias))
            rot_bo_deg = rot_bo_rad*180./np.pi
            xo2 = xo - (v_spacing/8.+0.01)*np.cos(rot_bo_rad)
            yo2 = yo - (v_spacing/8.+0.01)*np.sin(rot_bo_rad)
            xo1 = xo2 -0.05 *np.cos(rot_bo_rad)
            yo1 = yo2 -0.05 *np.sin(rot_bo_rad)
    # Output-Arrows
    layer_top_0 = v_spacing*(layer_sizes[-1] - 1)/2. + (top + bottom)/2.
    for m in range(layer_sizes[-1]):
        plt.arrow(right+0.015, layer_top_0 - m*v_spacing, 0.16*h_spacing, 0,  lw =1, head_width=0.01, head_length=0.02)
    plt.savefig("nn.png",bbox_inches='tight',pad_inches=0)
    plt.cla()
    plt.clf()

st.markdown(
        f"""
<style>
    .sidebar-content {{padding: 1rem !important;background: white !important;color:black;}}
</style>
""",
unsafe_allow_html=True)
st.markdown("""<h1 style='color:#292929;text-align: center;margin-top:-50px;font-size:50px;'>Neural<span style='text-align:center;color:rgb(246, 51, 102);'>Viz</span>.</h1><h3 style='text-align: center;margin-top:-25px;'>Created by: <a href='https://in.linkedin.com/in/rohankokkula'><b>Rohan Kokkula.<a href="https://in.linkedin.com/in/rohankokkula" target="_blank">
  <img align="right" alt="Rohan Kokkula | Twitter" width="22px" src="https://cdn.jsdelivr.net/npm/simple-icons@v3/icons/linkedin.svg" />
</a></b></a></h3>""", unsafe_allow_html=True)
options=st.sidebar.selectbox("",("Neural Visualizer","Cost Visualizer",("Back Propagation step-wise")))
if(options=="Neural Visualizer"):
    inps=st.sidebar.slider("Enter no. of Input Neurons.",2,100,2)
    hl=st.sidebar.slider("Enter no. of hidden Layers.",1,50,1)
    hls=[]
    for i in range(hl):
        hids=st.sidebar.slider(f"Enter no. of hidden Neurons for H{i+1} Layer",1,20,1)
        hls.append(hids)
    outs=st.sidebar.slider("Enter no. of output Neurons.",1,10,2)
    final=[inps]
    for j in hls:
        final.append(j)
    final.append(outs)        
    draw_neural_net(final,12,0.4,not st.sidebar.checkbox("Hide weights"))
    st.image('nn.png',use_column_width=True)
elif(options=="Cost Visualizer"):
    inps=st.sidebar.slider("Enter no. of Input Neurons.",1,10,2)
    outs=st.sidebar.slider("Enter no. of Output Neurons.",1,10,1)
    hids=st.sidebar.slider("Enter no. of Hidden Neurons.",1,10,3)
    draw_neural_net([inps,hids,outs],13,0.7,True)
    st.image('nn.png',use_column_width=True)
    np.random.seed(0)
    inpvals=np.random.choice([0,1], size=(inps*inps,),replace=True).tolist()
    c1=st.multiselect('Select Input values', inpvals)
    if(len(c1)==inps):
        for j in range(len(c1)):
            st.text(f"X{j+1} = "+str(c1[j]))
        hwts=np.random.uniform(low=0.1, high=2, size=(hids*inps,)).round(2).tolist()
        st.text("weights between input-hidden layer:")
        latest=np.array(hwts).reshape(hids,inps)
        for b,g in enumerate(latest):
            for u in range(inps):
                st.text(f"Wih_{u+1}{b+1}: {g[u]}")
        outvals=np.random.choice(np.arange(inps), size=(hids*outs,),replace=True).tolist()
        c2=st.multiselect('Select Output values', outvals)
        if(len(c2)==outs):
            for m in range(len(c2)):
                st.text(f"Y{m+1} = "+str(c2[m]))
            owts=np.random.uniform(low=0.1, high=2, size=(hids*outs,)).round(2).tolist()
            st.text("weights between hidden-output layer:")
            late=np.array(owts).reshape(outs,hids)
            for b,d in enumerate(late):
                for u in range(hids):
                    st.text(f"Who_{u+1}{b+1}: {d[u]}")
            rate=st.sidebar.slider("Learning rate",0.001,100.0,0.05)
            eps=st.sidebar.number_input("No. of Epochs.",1,10000,1)
            if(st.checkbox("Early Stopping")):
                early=st.slider("Select Early stopping threshold",0.0,1.0,0.05)
            else:
                early=False
            slot2=st.empty()
            slot=st.empty()
            nn = NeuralNetwork(inps, hids, outs, hidden_layer_weights=hwts, hidden_layer_bias=0.40, output_layer_weights=owts, output_layer_bias=0.40,lr=rate)
            losses=[]
            df=pd.DataFrame(losses,columns=['L2 LOSS'])
            for i in range(1,eps+1):
                nn.train(c1, c2)
                loss=nn.calculate_total_error([[c1,c2]])
                if(loss<=early):
                    break
                df.loc[i]=loss
                slot2.line_chart(df,use_container_width=True)
                slot.markdown(f"# Epoch: {(i)} Loss: {loss: 0.11f}")
else:
    st.empty()
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)