import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Initialize an empty dataframe for storing user data if it doesn't exist
if 'fitness_data' not in st.session_state:
    st.session_state.fitness_data = pd.DataFrame(columns=["user", "date", "age", "weight", "height", "session_duration", "total_steps", "calories_burned", "weight_loss", "calories_consumed"])

def clean_data(df):
    # Data Cleaning
    df = df.drop_duplicates()
    df["calories_burned"] = df["calories_burned"].fillna(0)
    df["weight_loss"] = df["weight_loss"].fillna(0)
    df["calories_consumed"] = df["calories_consumed"].fillna(df["calories_consumed"].mean())

    numeric_cols = ["age", "weight", "height", "session_duration", "total_steps", "calories_burned", "weight_loss", "calories_consumed"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")
    return df

st.title(":rainbow[Fitness Tracker]")
st.sidebar.title(":rainbow[My Fitness]")

menu = st.sidebar.radio("", ["Sign Up", "Log In", "Dashboard", "Predict My Calories", "Reports", "Fitness record"])

# Initialize the user credentials storage
if 'users' not in st.session_state:
    st.session_state.users = {}

# Sign Up Page
if menu == "Sign Up":
    st.image("log.png", width=90, channels="RBG")
    st.subheader("Create Your Account")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        if email not in st.session_state.users:
            st.session_state.users[email] = password
            st.success("Account created! Please log in.")
        else:
            st.warning("This email is already registered. Please log in.")

# Log In Page
elif menu == "Log In":
    st.image("log.png", width=90, channels="RBG")
    st.subheader("Log In to Your Account")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Log In"):
        if email in st.session_state.users and st.session_state.users[email] == password:
            st.success(f"Welcome back, {email}!")
            st.session_state["user"] = email
        else:
            st.error("Invalid credentials. Please try again.")

# Dashboard Page
elif menu == "Dashboard":
    if "user" not in st.session_state:
        st.warning("Please log in to access your dashboard.")
    else:
        user = st.session_state["user"]
        aa, bb, cc, dd, ee, ff, gg, uu = st.columns(8)
        if uu.button("log out"):
            if "user" in st.session_state:
                del st.session_state["user"]

        st.header(f"Welcome to Your Dashboard, {user}!")

        # Filter only the logged-in user's data
        user_data = st.session_state.fitness_data[st.session_state.fitness_data["user"] == user]

        # Add new fitness record
        st.subheader(":orange[Add New Fitness Record]")
        date = st.date_input("Date")
        age = st.number_input("Age", min_value=15, max_value=80, value=30)
        weight = st.number_input("Weight (kg)", min_value=40.0, max_value=200.0, value=70.0)
        height = st.number_input("Height (m)", min_value=1.2, max_value=2.5, value=1.7)
        session_duration = st.slider("Session Duration (hours)", min_value=0.5, max_value=3.0, value=1.0)
        total_steps = st.number_input("Total Steps", min_value=0, max_value=50000, value=5000)
        calories_intake = st.number_input("Calories Consumed", min_value=0, max_value=50000, value=0)

        # Calculate calories burned and weight loss
        total_calories = total_steps * 0.05
        cal = np.round(total_calories, 1)
        calories_deficit = cal - calories_intake
        pounds = calories_deficit / 3500
        weight_loss = pounds * 0.45
        w = np.round(weight_loss, 3)

        if st.button("Save"):
            new_record = {
                "user": user,
                "date": str(date),
                "age": age,
                "weight": weight,
                "height": height,
                "session_duration": session_duration,
                "total_steps": total_steps,
                "calories_burned": cal,
                "weight_loss": w,
                "calories_consumed": calories_intake,
            }
            # Add the new record to the session DataFrame
            st.session_state.fitness_data = pd.concat([st.session_state.fitness_data, pd.DataFrame([new_record])], ignore_index=True)
            st.success(f"Record saved!")
            st.header(f"Estimated Calories Burned: {cal}")
            st.header(f"Estimated weight loss: {w}")

        # Display calorie burn trend
        if not user_data.empty:
            st.subheader(":orange[Calories Burned Trend]")
            user_data["date"] = pd.to_datetime(user_data["date"])
            user_data = user_data.sort_values("date")
            st.area_chart(user_data.set_index("date")["calories_burned"], color=["#FF0000"])

# Predict My Calories Page
elif menu == "Predict My Calories":
    st.header("Calorie Burn Prediction")

    if "user" not in st.session_state:
        st.warning("Please log in to use this feature.")
    else:
        user = st.session_state["user"]

        # Fetch the logged-in user's fitness data for model training
        user_data = st.session_state.fitness_data[st.session_state.fitness_data["user"] == user]

        if not user_data.empty:
            df = clean_data(user_data)

            # Prepare features and target
            x = df[["session_duration", "total_steps"]]
            y = df["calories_burned"]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            # Train LinearRegression Model
            model = LinearRegression()
            model.fit(x_train, y_train)
            if st.button("Predict calories"):
                prediction = model.predict(x_test)
                p = np.round(prediction, 2)
                st.success(f"You are estimated to burn *{p}  calories* in your session!")
                mae = mean_absolute_error(y_test, prediction)
                r2 = r2_score(y_test, prediction)
                mse = mean_squared_error(y_test, prediction)
                st.write("mean absolute error : ", mae)
                st.write("r2 score : ", r2)
                st.write("mean square error : ", mse)

            n = df[["calories_consumed", "calories_burned"]]
            m = df["weight_loss"]

            n_train, n_test, m_train, m_test = train_test_split(n, m, test_size=0.2)

            models = LinearRegression()
            models.fit(n_train, m_train)
            st.header("Weight loss Prediction")
            if st.button("Predict weightloss"):
                pred = models.predict(n_test)
                pp = np.round(pred, 2)
                st.success(f"You are estimated to weight loss *{pp}  calories* in your session!")
                mae = mean_absolute_error(m_test, pred)
                r2 = r2_score(m_test, pred)
                mse = mean_squared_error(m_test, pred)
                st.write("mean absolute error : ", mae)
                st.write("r2 score : ", r2)
                st.write("mean square error : ", mse)
        else:
            st.warning("Not enough data for prediction. Add more records in the dashboard.")

elif menu == "Reports":
    if "user" not in st.session_state:
        st.warning("Please log in to access your dashboard.")
    else:
        user = st.session_state["user"]
        user_data = st.session_state.fitness_data[st.session_state.fitness_data["user"] == user]
        if not user_data.empty:
            df = clean_data(user_data)
            a, b, g, c = st.columns(4)
            a.metric(label="Total Steps", value=sum(df['total_steps']))
            b.metric(label="Total Calories Burned", value=sum(df['calories_burned']))
            g.metric(label="Weight Loss", value=sum(df["weight_loss"]))
            c.metric(label="Time (hours)", value=sum(df["session_duration"]))

            st.subheader("Calories Burned Trend")
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            st.line_chart(df.set_index("date")["calories_burned"], color=["#FF0000"])

            st.bar_chart(df,
                         x="date",
                         y=["calories_burned", "calories_consumed"],
                         color=["#FF0000", "#0000FF"],
            )
        else:
            st.info("No reports available. Please add fitness records.")

elif menu == "Fitness record":
    if "user" not in st.session_state:
        st.warning("Please log in to access your dashboard.")
    else:
        user = st.session_state["user"]
        user_data = st.session_state.fitness_data[st.session_state.fitness_data["user"] == user]
        
        if not user_data.empty:
            st.write("Your Fitness Records:", user_data)
            csv_data = user_data.to_csv(index=False)
            st.download_button(
                label="Download Fitness Records",
                data=csv_data,
                file_name=f"{user}_fitness_records.csv",
                mime="text/csv"
            )
        else:
            st.warning("No fitness records available for download.")
