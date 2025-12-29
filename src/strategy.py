import pandas as pd

def generate_action_plan(merged_df):
    """
    Decides actions by comparing Attribution (Volume) vs MMM (Efficiency).
    Expects columns: 'channel', 'attr_weight', 'mmm_roi'.
    """
    recommendations = []
    df = merged_df.copy()

    for _, row in df.iterrows():
        channel = row.get('channel', 'Unknown')
        vol = float(row.get('attr_weight', 0))
        roi = float(row.get('mmm_roi', 0))
        
        # Decision Matrix
        if vol > 0.15 and roi > 1.2:
            action, color, reason = "SCALE", "text-green-600", "High impact and high efficiency."
        elif vol > 0.15 and roi < 0.9:
            action, color, reason = "OPTIMIZE", "text-blue-600", "Driving volume but expensive. Refresh creative."
        elif vol < 0.05 and roi > 1.5:
            action, color, reason = "TEST", "text-emerald-500", "Small scale but very efficient. Increase spend."
        elif vol < 0.05 and roi < 0.7:
            action, color, reason = "CUT", "text-red-500", "Low volume and negative ROI."
        else:
            action, color, reason = "MAINTAIN", "text-slate-500", "Performing within expected bounds."

        recommendations.append({
            "ch": channel,
            "markov": f"{round(vol * 100, 1)}%",
            "roi": f"{round(roi, 2)}x",
            "action": action,
            "color": color,
            "reason": reason
        })
    return recommendations