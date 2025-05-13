#Assisted using AI
# Cultural Consensus Theory (CCT) Implementation using PyMC

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

def load_plant_knowledge_data(filepath):
    """
    Load the plant knowledge dataset and return it as a numpy array.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing the plant knowledge data
        
    Returns:
    --------
    numpy.ndarray
        Array of binary responses (0/1) with shape (N informants, M questions)
    """
    df = pd.read_csv(filepath)
    # Removes the 'informant' column
    data_matrix = df.iloc[:, 1:].values
    return data_matrix

def run_cct_model(data):
    """
    Implement the Cultural Consensus Theory model using PyMC.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Binary response data with shape (N informants, M questions)
        
    Returns:
    --------
    dict
        Dictionary containing the trace, summary, and other model outputs
    """
    N, M = data.shape  # N informants, M questions
    
    with pm.Model() as cct_model:
        # Define priors
        
        # Prior for informant competence (D)
        # We use a Beta (2, 1) prior to favor higher competence values while allowing the full range from 0.5 (guessing) to 1.0 (perfect knowledge)
        # The shifted Beta distribution ensures Di >= 0.5
        D_raw = pm.Beta("D_raw", alpha=2, beta=1, shape=N) 
        D = pm.Deterministic("D", 0.5 + 0.5 * D_raw)
        
        # Prior for consensus answers (Z)
        # We use Bernoulli (0.5) as a non-informative prior for each item
        Z = pm.Bernoulli("Z", 0.5, shape=M)
        
        # Calculate probability of giving a "1" response
        # We reshape D to broadcast correctly with Z
        D_reshaped = D[:, None]  # Shape: (N, 1)
        
        # p_ij = Z_j * D_i + (1 - Z_j) * (1 - D_i)
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)
        
        # Define likelihood
        X = pm.Bernoulli("X", p=p, observed=data)
        
        # Perform inference
        trace = pm.sample(
            2000,           # Number of draws
            tune=1000,      # Number of tuning steps
            chains=4,       # Number of chains
            random_seed=42  # For reproducibility
        )
        
        # Get summary statistics
        summary = az.summary(trace)
        
    # Return results
    return {
        "trace": trace,
        "summary": summary,
        "model": cct_model
    }

def analyze_results(results, data):
    """
    Analyze the results of the CCT model.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    data : numpy.ndarray
        Original data matrix
        
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    trace = results["trace"]
    summary = results["summary"]
    
    # Extract posterior samples
    posterior_samples = trace.posterior
    
    # Estimate informant competence
    competence_means = posterior_samples["D"].mean(dim=["chain", "draw"]).values
    
    # Get consensus answers
    z_means = posterior_samples["Z"].mean(dim=["chain", "draw"]).values
    consensus_answers = np.round(z_means).astype(int)
    
    # Calculate majority vote answers
    majority_vote = np.round(data.mean(axis=0)).astype(int)
    
    # Compare consensus with majority vote
    agreement = np.mean(consensus_answers == majority_vote) * 100
    
    # Return analysis results
    return {
        "competence_means": competence_means,
        "z_means": z_means,
        "consensus_answers": consensus_answers,
        "majority_vote": majority_vote,
        "agreement_percentage": agreement
    }

def visualize_results(results, analysis):
    """
    Create visualizations of the CCT model results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing model results
    analysis : dict
        Dictionary containing analysis results
        
    Returns:
    --------
    dict
        Dictionary containing plots
    """
    trace = results["trace"]
    
    # Create plots
    plots = {}
    
    # Plot convergence diagnostics - trace plots
    plots["trace_plot"] = az.plot_trace(trace, var_names=["D", "Z"])
    
    # Plot posterior distributions for competence
    plots["competence_posterior"] = az.plot_posterior(trace, var_names=["D"], hdi_prob=0.95)
    
    # Plot posterior distributions for consensus answers
    plots["z_posterior"] = az.plot_posterior(trace, var_names=["Z"], hdi_prob=0.95)
    
    # Plot competence estimates with CI
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get competence values and credible intervals
    competence_data = []
    for i in range(len(analysis["competence_means"])):
        competence_samples = trace.posterior["D"][:, :, i].values.flatten()
        competence_data.append({
            "informant": f"P{i+1}",
            "mean": competence_samples.mean(),
            "lower": np.percentile(competence_samples, 2.5),
            "upper": np.percentile(competence_samples, 97.5)
        })
    
    # Sort by mean competence
    competence_data.sort(key=lambda x: x["mean"], reverse=True)
    
    # Plot
    informants = [d["informant"] for d in competence_data]
    means = [d["mean"] for d in competence_data]
    lower = [d["lower"] for d in competence_data]
    upper = [d["upper"] for d in competence_data]
    
    ax.errorbar(means, informants, xerr=[np.array(means) - np.array(lower), np.array(upper) - np.array(means)], 
                fmt='o', capsize=5, ecolor='black', markersize=8)
    ax.set_title("Estimated Informant Competence with 95% CI")
    ax.set_xlabel("Competence")
    ax.set_ylabel("Informant")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0.4, 1.0)
    plots["competence_ranking"] = fig
    
    # Compare consensus vs majority vote
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(analysis["consensus_answers"]))
    width = 0.35
    
    ax.bar(x - width/2, analysis["consensus_answers"], width, label='CCT Consensus')
    ax.bar(x + width/2, analysis["majority_vote"], width, label='Majority Vote')
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}" for i in range(len(analysis["consensus_answers"]))])
    ax.set_ylabel('Answer (0/1)')
    ax.set_title('Consensus vs Majority Vote Answers')
    ax.legend()
    plots["consensus_vs_majority"] = fig
    
    # Return plots
    return plots

def main():
    # Load data
    # Try different possible file paths based on how the directory structure might be set up
    try:
        data = load_plant_knowledge_data("/home/jovyan/cct-midterm/cct-midterm/data/plant_knowledge.csv")
    except FileNotFoundError:
        try:
            data = load_plant_knowledge_data("../data/plant_knowledge.csv")
        except FileNotFoundError:
            data = load_plant_knowledge_data("../../data/plant_knowledge.csv")
    
    # Run CCT model
    results = run_cct_model(data)
    
    # Analyze results
    analysis = analyze_results(results, data)
    
    # Visualize results
    plots = visualize_results(results, analysis)
    
    # Print summary report
    print("\n===== CULTURAL CONSENSUS THEORY ANALYSIS REPORT =====\n")
    
    # Check convergence
    r_hat_values = results["summary"]["r_hat"].values
    max_r_hat = np.max(r_hat_values[~np.isnan(r_hat_values)])
    print(f"Convergence check - Maximum R-hat value: {max_r_hat:.3f}")
    if max_r_hat < 1.05:
        print("The model has converged well (all R-hat values < 1.05).")
    else:
        print("Some parameters may not have converged fully (R-hat >= 1.05).")
    
    # Print competence estimates
    print("\nInformant Competence Estimates (sorted from highest to lowest):")
    competence_df = pd.DataFrame({
        "Informant": [f"P{i+1}" for i in range(len(analysis["competence_means"]))],
        "Competence": analysis["competence_means"]
    }).sort_values("Competence", ascending=False)
    print(competence_df)
    
    # Most and least competent informants
    most_competent = competence_df.iloc[0]["Informant"]
    least_competent = competence_df.iloc[-1]["Informant"]
    print(f"\nMost competent informant: {most_competent} (D = {competence_df.iloc[0]['Competence']:.3f})")
    print(f"Least competent informant: {least_competent} (D = {competence_df.iloc[-1]['Competence']:.3f})")
    
    # Print consensus answers
    print("\nConsensus Answers:")
    consensus_df = pd.DataFrame({
        "Question": [f"PQ{i+1}" for i in range(len(analysis["consensus_answers"]))],
        "Consensus": analysis["consensus_answers"],
        "Posterior Probability": analysis["z_means"],
        "Majority Vote": analysis["majority_vote"]
    })
    print(consensus_df)
    
    # Compare with majority vote
    disagree_indices = np.where(analysis["consensus_answers"] != analysis["majority_vote"])[0]
    print(f"\nAgreement between CCT consensus and majority vote: {analysis['agreement_percentage']:.1f}%")
    if len(disagree_indices) > 0:
        print(f"Questions where CCT consensus differs from majority vote: {', '.join([f'PQ{i+1}' for i in disagree_indices + 1])}")
    else:
        print("CCT consensus and majority vote agree on all questions.")
    
    # Save plots if desired
    # for name, fig in plots.items():
    #     plt.figure(fig.number)
    #     plt.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    
    plt.show()

if __name__ == "__main__":
    main()