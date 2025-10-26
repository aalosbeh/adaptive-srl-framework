"""
Synthetic Data Generator for Self-Regulated Learning Framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import pickle
from dataclasses import dataclass


@dataclass
class LearnerProfile:
    """Profile for a synthetic learner"""
    learner_id: int
    institution_id: int
    age: int
    education_level: str
    baseline_ability: float
    learning_style: str
    motivation_level: float
    metacognitive_maturity: float
    generation: str  # Generation Z, Millennial, etc.


class SyntheticDataGenerator:
    """Generate synthetic data for SRL experiments"""
    
    def __init__(
        self,
        num_institutions: int = 10,
        num_learners: int = 1000,
        num_days: int = 180,
        domains: List[str] = ['mathematics', 'science', 'programming', 'language_arts'],
        random_seed: int = 42
    ):
        """
        Initialize synthetic data generator
        
        Args:
            num_institutions: Number of educational institutions
            num_learners: Total number of learners
            num_days: Number of days to simulate
            domains: Learning domains
            random_seed: Random seed for reproducibility
        """
        self.num_institutions = num_institutions
        self.num_learners = num_learners
        self.num_days = num_days
        self.domains = domains
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        # Generate learner profiles
        self.learner_profiles = self._generate_learner_profiles()
        
    def _generate_learner_profiles(self) -> List[LearnerProfile]:
        """Generate learner profiles"""
        profiles = []
        
        learners_per_institution = self.num_learners // self.num_institutions
        
        for inst_id in range(self.num_institutions):
            for local_id in range(learners_per_institution):
                learner_id = inst_id * learners_per_institution + local_id
                
                # Randomly assign characteristics
                age = np.random.randint(18, 30)
                
                # Determine generation based on age (simplified)
                if age <= 25:
                    generation = "Generation Z"
                else:
                    generation = "Millennial"
                
                education_levels = ['high_school', 'undergraduate', 'graduate']
                education_level = np.random.choice(education_levels)
                
                baseline_ability = np.random.beta(5, 2)  # Skewed towards higher ability
                
                learning_styles = ['visual', 'auditory', 'kinesthetic', 'reading_writing']
                learning_style = np.random.choice(learning_styles)
                
                motivation_level = np.random.beta(3, 2)
                metacognitive_maturity = np.random.beta(2, 3)  # Skewed towards lower maturity
                
                profile = LearnerProfile(
                    learner_id=learner_id,
                    institution_id=inst_id,
                    age=age,
                    education_level=education_level,
                    baseline_ability=baseline_ability,
                    learning_style=learning_style,
                    motivation_level=motivation_level,
                    metacognitive_maturity=metacognitive_maturity,
                    generation=generation
                )
                
                profiles.append(profile)
        
        return profiles
    
    def _simulate_learning_session(
        self,
        profile: LearnerProfile,
        day: int,
        domain: str,
        current_knowledge: float
    ) -> Dict:
        """
        Simulate a single learning session
        
        Args:
            profile: Learner profile
            day: Current day
            domain: Learning domain
            current_knowledge: Current knowledge level
        
        Returns:
            Dictionary of session data
        """
        # Simulate metacognitive states
        # Awareness: understanding of one's own learning process
        awareness_base = profile.metacognitive_maturity
        awareness_noise = np.random.normal(0, 0.1)
        awareness = np.clip(awareness_base + awareness_noise, 0, 1)
        
        # Monitoring: tracking progress and understanding
        monitoring_base = profile.metacognitive_maturity * 0.9 + profile.motivation_level * 0.1
        monitoring_noise = np.random.normal(0, 0.1)
        monitoring = np.clip(monitoring_base + monitoring_noise, 0, 1)
        
        # Control: ability to regulate learning strategies
        control_base = profile.metacognitive_maturity * 0.8 + profile.baseline_ability * 0.2
        control_noise = np.random.normal(0, 0.1)
        control = np.clip(control_base + control_noise, 0, 1)
        
        # Simulate engagement
        engagement_base = profile.motivation_level * 0.7 + awareness * 0.3
        engagement_noise = np.random.normal(0, 0.15)
        engagement = np.clip(engagement_base + engagement_noise, 0, 1)
        
        # Simulate learning progress
        learning_rate = (
            profile.baseline_ability * 0.4 +
            control * 0.3 +
            engagement * 0.2 +
            monitoring * 0.1
        )
        
        knowledge_gain = learning_rate * (1 - current_knowledge) * 0.05  # Diminishing returns
        new_knowledge = np.clip(current_knowledge + knowledge_gain, 0, 1)
        
        # Generate multi-modal data
        session_data = {
            'learner_id': profile.learner_id,
            'institution_id': profile.institution_id,
            'day': day,
            'domain': domain,
            'generation': profile.generation,
            
            # Metacognitive states (ground truth)
            'awareness': awareness,
            'monitoring': monitoring,
            'control': control,
            
            # Other states
            'engagement': engagement,
            'knowledge_level': new_knowledge,
            'knowledge_gain': knowledge_gain,
            
            # Text data features (simulated from reflections)
            'text_sentiment': np.random.normal(0.6, 0.2),
            'text_metacog_markers': np.random.poisson(5),
            'text_length': np.random.randint(50, 500),
            
            # Visual data features (simulated from engagement)
            'visual_attention_score': engagement + np.random.normal(0, 0.1),
            'visual_emotion_valence': np.random.normal(0.5, 0.2),
            'visual_gaze_pattern': np.random.choice(['focused', 'scattered', 'wandering']),
            
            # Temporal data features
            'session_duration': np.random.gamma(2, 30),  # minutes
            'num_interactions': np.random.poisson(50),
            'pause_frequency': np.random.gamma(2, 5),
            
            # Graph data features (simulated social/concept connections)
            'social_connections': np.random.poisson(3),
            'concept_mastery': np.random.beta(2, 2, size=5),  # 5 related concepts
        }
        
        return session_data
    
    def generate(self) -> Dict[str, any]:
        """
        Generate complete synthetic dataset
        
        Returns:
            Dictionary containing all generated data
        """
        print(f"Generating synthetic data for {self.num_learners} learners...")
        print(f"Institutions: {self.num_institutions}, Days: {self.num_days}")
        
        all_sessions = []
        
        # Track knowledge levels for each learner-domain pair
        knowledge_tracker = {
            (profile.learner_id, domain): 0.1 + np.random.uniform(0, 0.2)
            for profile in self.learner_profiles
            for domain in self.domains
        }
        
        for day in range(self.num_days):
            if day % 30 == 0:
                print(f"Generating day {day}/{self.num_days}...")
            
            for profile in self.learner_profiles:
                # Each learner works on 1-2 domains per day
                num_domains = np.random.choice([1, 2], p=[0.6, 0.4])
                selected_domains = np.random.choice(self.domains, size=num_domains, replace=False)
                
                for domain in selected_domains:
                    current_knowledge = knowledge_tracker[(profile.learner_id, domain)]
                    
                    session_data = self._simulate_learning_session(
                        profile, day, domain, current_knowledge
                    )
                    
                    # Update knowledge tracker
                    knowledge_tracker[(profile.learner_id, domain)] = session_data['knowledge_level']
                    
                    all_sessions.append(session_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_sessions)
        
        # Organize by institution
        institutional_data = []
        for inst_id in range(self.num_institutions):
            inst_df = df[df['institution_id'] == inst_id].copy()
            institutional_data.append(inst_df)
        
        dataset = {
            'sessions': df,
            'institutional_data': institutional_data,
            'learner_profiles': self.learner_profiles,
            'metadata': {
                'num_institutions': self.num_institutions,
                'num_learners': self.num_learners,
                'num_days': self.num_days,
                'domains': self.domains,
                'total_sessions': len(df)
            }
        }
        
        print(f"Generated {len(df)} learning sessions")
        print(f"Average sessions per learner: {len(df) / self.num_learners:.1f}")
        
        return dataset
    
    def save(self, dataset: Dict, filepath: str):
        """Save dataset to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> Dict:
        """Load dataset from file"""
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Dataset loaded from {filepath}")
        return dataset
    
    def prepare_for_training(self, dataset: Dict) -> List[Dict]:
        """
        Prepare dataset for federated training
        
        Args:
            dataset: Generated dataset
        
        Returns:
            List of training data dictionaries for each institution
        """
        institutional_training_data = []
        
        for inst_id in range(self.num_institutions):
            inst_df = dataset['institutional_data'][inst_id]
            
            # Extract features for state representation
            states = inst_df[[
                'awareness', 'monitoring', 'control', 'engagement', 'knowledge_level',
                'text_sentiment', 'visual_attention_score', 'session_duration'
            ]].values
            
            # Simulate actions (interventions)
            # In real scenario, these would be actual interventions taken
            actions = np.random.randn(len(inst_df), 4)  # 4 intervention types
            
            # Compute rewards based on learning progress
            rewards = inst_df['knowledge_gain'].values * 10  # Scale rewards
            
            # Simulate log probabilities (would come from policy during actual training)
            log_probs = np.random.randn(len(inst_df))
            
            # Done flags (episode ends)
            dones = np.zeros(len(inst_df))
            dones[inst_df.groupby('learner_id').tail(1).index - inst_df.index[0]] = 1
            
            # Next states (shifted by 1)
            next_states = np.roll(states, -1, axis=0)
            next_states[-1] = states[-1]  # Last next_state is same as last state
            
            training_data = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'log_probs': log_probs,
                'dones': dones,
                'next_states': next_states
            }
            
            institutional_training_data.append(training_data)
        
        return institutional_training_data


if __name__ == '__main__':
    # Example usage
    generator = SyntheticDataGenerator(
        num_institutions=10,
        num_learners=1000,
        num_days=180
    )
    
    dataset = generator.generate()
    generator.save(dataset, 'data/synthetic_dataset.pkl')
    
    # Prepare for training
    training_data = generator.prepare_for_training(dataset)
    print(f"Prepared training data for {len(training_data)} institutions")

