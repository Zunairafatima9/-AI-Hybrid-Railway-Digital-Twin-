import matplotlib.pyplot as plt

# -------------------------------
# FIGURE 1: Error Distribution
# -------------------------------
error_ranges = ["<2 min", "2-5 min", "5-10 min", ">10 min"]
percentages = [100, 0, 0, 0]

plt.figure()
plt.bar(error_ranges, percentages)
plt.xlabel("Prediction Error Range")
plt.ylabel("Percentage of Predictions (%)")
plt.title("XGBoost Error Distribution")
plt.savefig("fig1_error_distribution.png", dpi=300, bbox_inches='tight')
plt.close()


# -------------------------------
# FIGURE 2: Model Performance Metrics
# -------------------------------
metrics = ["MAE", "RMSE", "R²"]
values = [0.332, 0.439, 0.605]

plt.figure()
plt.bar(metrics, values)
plt.ylabel("Metric Value")
plt.title("XGBoost Performance Metrics")
plt.savefig("fig2_performance_metrics.png", dpi=300, bbox_inches='tight')
plt.close()


# -------------------------------
# FIGURE 3: Feature Importance
# -------------------------------
features = [
    "excess_travel_time",
    "block_count",
    "num_tracks",
    "distance_km",
    "station_avg_delay",
    "to_station_enc",
    "hour_of_day",
    "from_station_enc",
    "dispatch_priority",
    "train_type_enc"
]

importance = [
    0.3583, 0.1412, 0.0871, 0.0848, 0.0768,
    0.0603, 0.0551, 0.0496, 0.0462, 0.0407
]

plt.figure()
plt.barh(features, importance)
plt.xlabel("Importance Score")
plt.title("Feature Importance (XGBoost)")
plt.gca().invert_yaxis()
plt.savefig("fig3_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()


# -------------------------------
# FIGURE 4: DQN vs Baselines
# -------------------------------
policies = ["DQN", "Random", "Always", "Never"]
scores = [4345, -2390, -8590, 4345]

plt.figure()
plt.bar(policies, scores)
plt.ylabel("Policy Score")
plt.title("DQN vs Baseline Policies")
plt.savefig("fig4_dqn_vs_baselines.png", dpi=300, bbox_inches='tight')
plt.close()


# -------------------------------
# FIGURE 5: Intervention Accuracy
# -------------------------------
models = ["DQN", "Random"]
accuracy = [91.4, 48.0]

plt.figure()
plt.bar(models, accuracy)
plt.ylabel("Accuracy (%)")
plt.title("Intervention Accuracy Comparison")
plt.savefig("fig5_accuracy_comparison.png", dpi=300, bbox_inches='tight')
plt.close()


print("All figures generated successfully.")