"""
State management for dashboard analysis.

This module handles the persistence and management of analysis state
across different dashboard components and visualization tabs.

Classes:
    AnalysisState: Manages state for analysis session.
"""

class AnalysisState:
    """State management for analysis session.
    
    Maintains the state of the current analysis session, including
    model outputs, analysis results, and visualization data.
    
    Attributes:
        hidden_states: Model hidden state tensors.
        attentions: Model attention tensors.
        tokens (list): Processed input tokens.
        results_df (pd.DataFrame): Analysis results.
        model_choice (str): Selected model identifier.
        text (str): Input text being analyzed.
        predictions (dict): Model predictions.
    """

    def __init__(self):
        self.hidden_states = None
        self.attentions = None
        self.tokens = None
        self.results_df = None
        self.model_choice = None
        self.text = None
        self.predictions = None

    def update(self, hidden_states, attentions, tokens, results_df, model_choice, text, predictions):
        self.hidden_states = hidden_states
        self.attentions = attentions
        self.tokens = tokens
        self.results_df = results_df
        self.model_choice = model_choice
        self.text = text
        self.predictions = predictions

    def is_initialized(self):
        # Check each component individually
        states_check = self.hidden_states is not None
        attention_check = self.attentions is not None
        tokens_check = self.tokens is not None
        df_check = self.results_df is not None and not self.results_df.empty
        predictions_check = self.predictions is not None
        
        return all([states_check, attention_check, tokens_check, df_check, predictions_check])
