import streamlit as st
import pandas as pd
import numpy as np
import pulp

st.set_page_config(page_title="C2C Matchmaker", layout="wide")
st.title("🏥 Connection to Care (C2C) Automated Matchmaker")

# Sidebar Controls
st.sidebar.header("Simulation Settings")
num_patients = st.sidebar.slider("Number of Patients", 10, 200, 100)
num_doctors = st.sidebar.slider("Number of Doctors", 5, 50, 25)
seed = st.sidebar.number_input("Random Seed", value=42)

if st.sidebar.button("Run Matchmaker Engine"):
    np.random.seed(seed)
    insurances = ['Aetna', 'Cigna', 'BlueCross', 'UnitedHealthcare', 'Medicare']

    # Generate Datasets
    patients = pd.DataFrame({
        'Patient_ID': [f"P_{i:03d}" for i in range(num_patients)],
        'Loc_X': np.random.uniform(0, 100, num_patients),
        'Loc_Y': np.random.uniform(0, 100, num_patients),
        'Insurance': np.random.choice(insurances, num_patients)
    })

    doctors = pd.DataFrame({
        'Doctor_ID': [f"D_{j:02d}" for j in range(num_doctors)],
        'Loc_X': np.random.uniform(0, 100, num_doctors),
        'Loc_Y': np.random.uniform(0, 100, num_doctors),
        'Accepted_Insurances': [
            list(np.random.choice(insurances, np.random.randint(2, 5), replace=False)) 
            for _ in range(num_doctors)
        ],
        'Capacity': np.random.randint(4, 8, num_doctors)
    })

    # Linear Optimization Core
    prob = pulp.LpProblem("C2C_Matching", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("match", ((i, j) for i in patients.index for j in doctors.index), cat='Binary')

    objective_terms = []
    for i in patients.index:
        for j in doctors.index:
            dist = np.sqrt((patients.loc[i, 'Loc_X'] - doctors.loc[j, 'Loc_X'])**2 + 
                           (patients.loc[i, 'Loc_Y'] - doctors.loc[j, 'Loc_Y'])**2)
            if patients.loc[i, 'Insurance'] in doctors.loc[j, 'Accepted_Insurances']:
                objective_terms.append(dist * x[i, j])
            else:
                prob += x[i, j] == 0

    prob += pulp.lpSum(objective_terms)

    for i in patients.index:
        prob += pulp.lpSum(x[i, j] for j in doctors.index) == 1

    for j in doctors.index:
        prob += pulp.lpSum(x[i, j] for i in patients.index) <= doctors.loc[j, 'Capacity']

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Display Output UI
    if pulp.LpStatus[prob.status] == 'Optimal':
        matches = []
        for i in patients.index:
            for j in doctors.index:
                if x[i, j].varValue == 1.0:
                    dist = np.sqrt((patients.loc[i, 'Loc_X'] - doctors.loc[j, 'Loc_X'])**2 + 
                                   (patients.loc[i, 'Loc_Y'] - doctors.loc[j, 'Loc_Y'])**2)
                    matches.append({
                        'Patient ID': patients.loc[i, 'Patient_ID'],
                        'Patient Insurance': patients.loc[i, 'Insurance'],
                        'Matched Doctor': doctors.loc[j, 'Doctor_ID'],
                        'Travel Distance': round(dist, 2)
                    })

        results_df = pd.DataFrame(matches)

        st.success(f"Matched {len(results_df)} patients successfully across {num_doctors} available doctors.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Match Results Table")
            st.dataframe(results_df, use_container_width=True)
        with col2:
            st.subheader("Geographic Assignment Map")
            st.scatter_chart(data=patients, x='Loc_X', y='Loc_Y', size=50)

    else:
        st.error("Matching failed: System capacity is lower than patient count or insurance coverage gaps exist.")
