import streamlit as st

def main():
    st.title("RobustML Test App")
    st.write("This is a minimal working Streamlit app.")
    
    name = st.text_input("Enter your name:")
    if name:
        st.success(f"Hello, {name}!")

if __name__ == "__main__":
    main()
