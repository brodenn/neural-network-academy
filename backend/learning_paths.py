"""
Learning Paths for Neural Network Academy

Defines structured curricula guiding learners through the 32 problems
with clear progression, prerequisites, and educational scaffolding.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Literal

# Step types for different interactive challenges
StepType = Literal['training', 'build_challenge', 'prediction_quiz', 'debug_challenge', 'tuning_challenge']

@dataclass
class BuildChallengeData:
    """Data for build_challenge step type"""
    min_layers: int = 0
    max_layers: int = 5
    min_hidden_neurons: int = 1
    max_hidden_neurons: int = 16
    must_have_hidden: bool = False
    success_message: str = "Great architecture!"

@dataclass
class PredictionQuizData:
    """Data for prediction_quiz step type"""
    quiz_id: str  # References PREDICTION_QUIZZES in frontend
    show_result_after: bool = True  # Show training result after quiz

@dataclass
class DebugChallengeData:
    """Data for debug_challenge step type"""
    challenge_id: str  # References DEBUG_CHALLENGES in frontend

@dataclass
class TuningChallengeData:
    """Data for tuning_challenge step type - find optimal hyperparameters"""
    parameter: str  # 'learning_rate', 'epochs', 'architecture'
    target_loss: float = 0.1
    max_attempts: int = 5

@dataclass
class PathStep:
    step_number: int
    problem_id: str
    title: str
    learning_objectives: List[str]
    required_accuracy: float = 0.95
    hints: List[str] = None
    # New: step type for interactive challenges
    step_type: StepType = 'training'
    # Challenge-specific data
    build_challenge: Optional[BuildChallengeData] = None
    prediction_quiz: Optional[PredictionQuizData] = None
    debug_challenge: Optional[DebugChallengeData] = None
    tuning_challenge: Optional[TuningChallengeData] = None

    def __post_init__(self):
        if self.hints is None:
            self.hints = []

@dataclass
class LearningPath:
    id: str
    name: str
    description: str
    difficulty: str  # 'beginner', 'intermediate', 'advanced', 'research'
    estimated_time: str
    badge_icon: str
    badge_title: str
    badge_color: str
    prerequisites: List[str]  # List of path IDs that must be completed first
    steps: List[PathStep]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'difficulty': self.difficulty,
            'estimatedTime': self.estimated_time,
            'prerequisites': self.prerequisites,
            'steps': len(self.steps),
            'badge': {
                'icon': self.badge_icon,
                'title': self.badge_title,
                'color': self.badge_color
            }
        }

    def get_step_details(self) -> List[Dict]:
        """Get detailed information about all steps"""
        result = []
        for step in self.steps:
            step_dict = {
                'stepNumber': step.step_number,
                'problemId': step.problem_id,
                'title': step.title,
                'learningObjectives': step.learning_objectives,
                'requiredAccuracy': step.required_accuracy,
                'hints': step.hints,
                'stepType': step.step_type,
            }
            # Add challenge-specific data if present
            if step.build_challenge:
                step_dict['buildChallenge'] = {
                    'minLayers': step.build_challenge.min_layers,
                    'maxLayers': step.build_challenge.max_layers,
                    'minHiddenNeurons': step.build_challenge.min_hidden_neurons,
                    'maxHiddenNeurons': step.build_challenge.max_hidden_neurons,
                    'mustHaveHidden': step.build_challenge.must_have_hidden,
                    'successMessage': step.build_challenge.success_message,
                }
            if step.prediction_quiz:
                step_dict['predictionQuiz'] = {
                    'quizId': step.prediction_quiz.quiz_id,
                    'showResultAfter': step.prediction_quiz.show_result_after,
                }
            if step.debug_challenge:
                step_dict['debugChallenge'] = {
                    'challengeId': step.debug_challenge.challenge_id,
                }
            if step.tuning_challenge:
                step_dict['tuningChallenge'] = {
                    'parameter': step.tuning_challenge.parameter,
                    'targetLoss': step.tuning_challenge.target_loss,
                    'maxAttempts': step.tuning_challenge.max_attempts,
                }
            result.append(step_dict)
        return result


# Define all learning paths

LEARNING_PATHS = {
    'foundations': LearningPath(
        id='foundations',
        name='Foundations',
        description='Master basic neural network concepts',
        difficulty='beginner',
        estimated_time='2-3 hours',
        badge_icon='🏆',
        badge_title='Foundation Scholar',
        badge_color='#3B82F6',
        prerequisites=[],
        steps=[
            PathStep(
                step_number=1,
                problem_id='and_gate',
                title='AND Gate - Single Neuron Capability',
                learning_objectives=[
                    'Understand what a single neuron can learn',
                    'See linear separability in action',
                    'Learn about activation functions'
                ],
                hints=[
                    'A single neuron can learn linearly separable patterns',
                    'Try training with just a few epochs to see quick convergence'
                ]
            ),
            PathStep(
                step_number=2,
                problem_id='or_gate',
                title='OR Gate - Linear Classification',
                learning_objectives=[
                    'Recognize another linearly separable problem',
                    'Compare with AND gate behavior',
                    'Understand decision boundaries'
                ],
                hints=[
                    'OR is also linearly separable like AND',
                    'Notice how similar the training is to AND'
                ]
            ),
            PathStep(
                step_number=3,
                problem_id='not_gate',
                title='NOT Gate - Simple Transformation',
                learning_objectives=[
                    'See how neurons invert signals',
                    'Understand bias importance',
                    'Learn about single-input neurons'
                ],
                hints=[
                    'This is the simplest possible neural network',
                    'The bias term is crucial for this transformation'
                ]
            ),
            PathStep(
                step_number=4,
                problem_id='nand_gate',
                title='NAND Gate - The Universal Gate',
                learning_objectives=[
                    'Learn about NAND as a universal gate',
                    'See another linearly separable problem',
                    'Understand that NAND can build ANY logic circuit'
                ],
                hints=[
                    'NAND is called "universal" - you can build ANY logic from it!',
                    'Still solvable with a single neuron like AND/OR'
                ]
            ),
            PathStep(
                step_number=5,
                problem_id='xor',
                title='XOR Gate - Why Hidden Layers Matter',
                learning_objectives=[
                    'Understand that XOR is NOT linearly separable',
                    'Learn why hidden layers are necessary',
                    'See how architecture affects what networks can learn'
                ],
                hints=[
                    'XOR cannot be solved with a single neuron!',
                    'The default [2, 4, 1] architecture adds hidden layers',
                    'Hidden layers allow the network to learn non-linear patterns'
                ]
            ),
            PathStep(
                step_number=6,
                problem_id='xnor',
                title='XNOR Gate - Mastering Hidden Layers',
                learning_objectives=[
                    'Apply hidden layer knowledge to XNOR',
                    'Recognize XNOR is similar to XOR',
                    'Build confidence with non-linear problems'
                ],
                hints=[
                    'XNOR is the opposite of XOR (1 when inputs match)',
                    'Same architecture as XOR should work well'
                ]
            ),
            PathStep(
                step_number=7,
                problem_id='xor_5bit',
                title='5-Bit Parity - Scaling Up',
                learning_objectives=[
                    'Apply XOR concept to 5 inputs instead of 2',
                    'See how networks scale to more complex problems',
                    'Learn about the parity problem'
                ],
                hints=[
                    'Parity checks if the sum of bits is odd or even',
                    'This is XOR extended to 5 inputs!',
                    'Deeper networks help with this complexity'
                ]
            )
        ]
    ),

    'deep-learning-basics': LearningPath(
        id='deep-learning-basics',
        name='Deep Learning Basics',
        description='Master training, initialization, and hyperparameters',
        difficulty='intermediate',
        estimated_time='4-5 hours',
        badge_icon='🧠',
        badge_title='Neural Navigator',
        badge_color='#8B5CF6',
        prerequisites=['foundations'],
        steps=[
            PathStep(
                step_number=1,
                problem_id='fail_zero_init',
                title='FAIL: Zero Initialization',
                learning_objectives=[
                    'Understand the symmetry breaking problem',
                    'Learn why initialization matters',
                    'See what happens without random weights'
                ],
                required_accuracy=0.0,
                hints=[
                    'All neurons learn the same thing with zero init!',
                    'Random initialization breaks symmetry',
                    'This is a fundamental requirement for learning'
                ]
            ),
            PathStep(
                step_number=2,
                problem_id='fail_lr_high',
                title='FAIL: Learning Rate Too High',
                learning_objectives=[
                    'Understand divergence and instability',
                    'Learn about learning rate importance',
                    'See what happens when updates are too large'
                ],
                required_accuracy=0.0,
                hints=[
                    'The loss will explode or oscillate wildly',
                    'Learning rate controls step size',
                    'Too high = overshooting the minimum'
                ]
            ),
            PathStep(
                step_number=3,
                problem_id='fail_lr_low',
                title='FAIL: Learning Rate Too Low',
                learning_objectives=[
                    'Understand slow convergence',
                    'Learn about optimization speed',
                    'See the tradeoff between stability and speed'
                ],
                required_accuracy=0.0,
                hints=[
                    'Training will be extremely slow',
                    'The network barely learns each epoch',
                    'Too low = taking tiny steps toward minimum'
                ]
            ),
            PathStep(
                step_number=4,
                problem_id='xor_5bit',
                title='5-bit Parity - Complex XOR',
                learning_objectives=[
                    'Scale up to more complex XOR problems',
                    'Understand how complexity grows',
                    'Learn about network capacity needs'
                ],
                hints=[
                    'This is XOR with 5 inputs instead of 2',
                    'You may need a larger network',
                    'The principle is the same, just more complex'
                ]
            ),
            PathStep(
                step_number=5,
                problem_id='linear',
                title='Linear Regression - Continuous Outputs',
                learning_objectives=[
                    'Introduction to regression problems',
                    'Learn about continuous vs discrete outputs',
                    'Understand linear function approximation'
                ],
                hints=[
                    'The network outputs a continuous value, not a class',
                    'This is the simplest regression problem',
                    'Linear relationships are easiest to learn'
                ]
            ),
            PathStep(
                step_number=6,
                problem_id='sine_wave',
                title='Sine Wave - Non-linear Regression',
                learning_objectives=[
                    'Learn about regression vs classification',
                    'Understand continuous outputs',
                    'See how networks approximate functions'
                ],
                hints=[
                    'The network learns to approximate a sine wave',
                    'Try different input values to see the curve',
                    'This is function approximation'
                ]
            ),
            PathStep(
                step_number=7,
                problem_id='polynomial',
                title='Polynomial - Non-linear Regression',
                learning_objectives=[
                    'Master non-linear function approximation',
                    'Understand different regression complexities',
                    'Learn about universal approximation'
                ],
                hints=[
                    'Neural networks can approximate any continuous function',
                    'Polynomials are good test cases',
                    'Watch how well the network fits the curve'
                ]
            ),
            PathStep(
                step_number=8,
                problem_id='two_blobs',
                title='Two Blobs - 2D Classification Intro',
                learning_objectives=[
                    'Introduction to 2D decision boundaries',
                    'Visualize how networks separate data in 2D',
                    'Build intuition before complex shapes'
                ],
                hints=[
                    'Two clusters that are easy to separate',
                    'Watch the decision boundary form',
                    'This is a warm-up for harder 2D problems'
                ]
            ),
            PathStep(
                step_number=9,
                problem_id='moons',
                title='Moons - Complex Boundaries',
                learning_objectives=[
                    'Handle complex non-linear boundaries',
                    'Understand curved decision surfaces',
                    'See advanced boundary shapes'
                ],
                hints=[
                    'The two "moons" interlock in a complex way',
                    'The decision boundary must curve around them',
                    'This tests the network\'s flexibility'
                ]
            ),
            PathStep(
                step_number=10,
                problem_id='circle',
                title='Circle - Radial Patterns',
                learning_objectives=[
                    'Learn about radial decision boundaries',
                    'Understand when linear separation fails',
                    'Master circular/elliptical boundaries'
                ],
                hints=[
                    'One class is inside a circle, one outside',
                    'No straight line can separate these',
                    'The network must learn a circular boundary'
                ]
            )
        ]
    ),

    'multi-class-mastery': LearningPath(
        id='multi-class-mastery',
        name='Multi-Class Mastery',
        description='Handle multiple output classes',
        difficulty='intermediate',
        estimated_time='2-3 hours',
        badge_icon='🎨',
        badge_title='Classifier Champion',
        badge_color='#8B5CF6',
        prerequisites=['foundations'],
        steps=[
            PathStep(
                step_number=1,
                problem_id='quadrants',
                title='Quadrant Classification - 4 Classes',
                learning_objectives=[
                    'Learn about multi-class classification',
                    'Understand softmax activation',
                    'See one-hot encoding in action'
                ],
                hints=[
                    'Each quadrant is a different class',
                    'The network outputs probabilities for each class',
                    'Softmax ensures outputs sum to 1'
                ]
            ),
            PathStep(
                step_number=2,
                problem_id='blobs',
                title='Gaussian Blobs - 5 Classes',
                learning_objectives=[
                    'Scale to more classes',
                    'Handle overlapping class regions',
                    'Learn about classification confidence'
                ],
                hints=[
                    '5 separate clusters in 2D space',
                    'Some regions may have lower confidence',
                    'The network learns probability distributions'
                ]
            ),
            PathStep(
                step_number=3,
                problem_id='patterns',
                title='Signal Patterns - Temporal Recognition',
                learning_objectives=[
                    'Recognize sequential patterns',
                    'Understand pattern classification',
                    'Learn about feature representation'
                ],
                hints=[
                    'Different signal patterns to classify',
                    'The network learns pattern features',
                    'This is like basic time-series classification'
                ]
            ),
            PathStep(
                step_number=4,
                problem_id='colors',
                title='Color Classification - RGB Mixing',
                learning_objectives=[
                    'Classify based on continuous features',
                    'Understand color space representation',
                    'Master 6-class problems'
                ],
                hints=[
                    '6 different color classes',
                    'RGB values as inputs',
                    'The network learns color boundaries'
                ]
            )
        ]
    ),

    'convolutional-vision': LearningPath(
        id='convolutional-vision',
        name='Convolutional Vision',
        description='Understand CNNs for spatial data',
        difficulty='advanced',
        estimated_time='3-4 hours',
        badge_icon='👁️',
        badge_title='Vision Virtuoso',
        badge_color='#F59E0B',
        prerequisites=['multi-class-mastery'],
        steps=[
            PathStep(
                step_number=1,
                problem_id='shapes',
                title='Shape Detection - Basic CNN',
                learning_objectives=[
                    'Understand convolutional layers',
                    'Learn about spatial feature extraction',
                    'See how CNNs process images'
                ],
                hints=[
                    'Draw shapes on the 8x8 grid',
                    'CNNs learn spatial patterns automatically',
                    'Watch the feature maps to see what it learns'
                ]
            ),
            PathStep(
                step_number=2,
                problem_id='digits',
                title='Digit Recognition - Advanced CNN',
                learning_objectives=[
                    'Master complex spatial patterns',
                    'Handle 10-class classification',
                    'Understand deep CNN architectures'
                ],
                hints=[
                    'Draw digits 0-9 on the grid',
                    'This is like mini-MNIST',
                    'CNNs excel at this task'
                ]
            ),
            PathStep(
                step_number=3,
                problem_id='patterns',
                title='Signal Pattern Classification',
                learning_objectives=[
                    'Apply CNNs to sequential data',
                    'Understand gesture recognition',
                    'Master advanced pattern recognition'
                ],
                hints=[
                    'Different gesture patterns',
                    'CNNs can handle time-series too',
                    'Feature extraction + classification'
                ]
            )
        ]
    ),

    'pitfall-prevention': LearningPath(
        id='pitfall-prevention',
        name='Pitfall Prevention',
        description='Learn what NOT to do',
        difficulty='intermediate',
        estimated_time='1-2 hours',
        badge_icon='🛡️',
        badge_title='Error Expert',
        badge_color='#EF4444',
        prerequisites=[],
        steps=[
            PathStep(
                step_number=1,
                problem_id='fail_xor_no_hidden',
                title='Insufficient Capacity',
                learning_objectives=[
                    'Understand capacity limitations',
                    'Learn when architecture matters',
                    'See fundamental limitations'
                ],
                required_accuracy=0.0
            ),
            PathStep(
                step_number=2,
                problem_id='fail_zero_init',
                title='Symmetry Problem',
                learning_objectives=[
                    'Understand initialization importance',
                    'Learn about symmetry breaking',
                    'See why randomness matters'
                ],
                required_accuracy=0.0
            ),
            PathStep(
                step_number=3,
                problem_id='fail_lr_high',
                title='Instability and Divergence',
                learning_objectives=[
                    'Understand hyperparameter sensitivity',
                    'Learn about numerical stability',
                    'See divergence in action'
                ],
                required_accuracy=0.0
            ),
            PathStep(
                step_number=4,
                problem_id='fail_lr_low',
                title='Slow Convergence',
                learning_objectives=[
                    'Understand optimization speed',
                    'Learn about efficiency tradeoffs',
                    'See practical training issues'
                ],
                required_accuracy=0.0
            ),
            PathStep(
                step_number=5,
                problem_id='fail_vanishing',
                title='Vanishing Gradients',
                learning_objectives=[
                    'Understand gradient flow issues',
                    'Learn about deep network problems',
                    'See why activation choice matters'
                ],
                required_accuracy=0.0
            ),
            PathStep(
                step_number=6,
                problem_id='fail_underfit',
                title='Underfitting',
                learning_objectives=[
                    'Understand model capacity',
                    'Learn about complexity matching',
                    'See when networks are too simple'
                ],
                required_accuracy=0.0
            )
        ]
    ),

    'research-frontier': LearningPath(
        id='research-frontier',
        name='Research Frontier',
        description='Tackle challenging problems',
        difficulty='advanced',
        estimated_time='4-5 hours',
        badge_icon='🚀',
        badge_title='Research Pioneer',
        badge_color='#EF4444',
        prerequisites=['deep-learning-basics', 'multi-class-mastery'],
        steps=[
            PathStep(
                step_number=1,
                problem_id='donut',
                title='Donut - Complex Topology',
                learning_objectives=[
                    'Handle complex topological patterns',
                    'Master non-convex boundaries',
                    'Understand advanced decision surfaces'
                ],
                hints=[
                    'One class forms a ring around another',
                    'This requires a complex boundary shape',
                    'Test your network\'s flexibility'
                ]
            ),
            PathStep(
                step_number=2,
                problem_id='spiral',
                title='Spiral - Highly Non-linear',
                learning_objectives=[
                    'Tackle extremely complex patterns',
                    'Learn about deep network requirements',
                    'Master challenging benchmarks'
                ],
                hints=[
                    'Two spirals intertwined',
                    'One of the hardest 2D problems',
                    'May need a deeper or wider network'
                ]
            ),
            PathStep(
                step_number=3,
                problem_id='surface',
                title='2D Surface - Multi-variable Regression',
                learning_objectives=[
                    'Handle multi-dimensional regression',
                    'Learn about surface approximation',
                    'Master complex function fitting'
                ],
                hints=[
                    'The network learns a 2D surface',
                    'Two inputs, one continuous output',
                    'Universal approximation in action'
                ]
            ),
            PathStep(
                step_number=4,
                problem_id='digits',
                title='Digit Recognition - 10-class CNN',
                learning_objectives=[
                    'Master complex CNN tasks',
                    'Handle many classes efficiently',
                    'Understand production-ready models'
                ],
                hints=[
                    'This is close to real-world problems',
                    'CNNs excel at image classification',
                    'Feature extraction is key'
                ]
            )
        ]
    ),

    # NEW: Interactive path with build challenges, prediction quizzes, and debug challenges
    'interactive-fundamentals': LearningPath(
        id='interactive-fundamentals',
        name='Interactive Fundamentals',
        description='Learn by DOING - build networks, predict outcomes, and debug problems',
        difficulty='beginner',
        estimated_time='1-2 hours',
        badge_icon='🎮',
        badge_title='Active Learner',
        badge_color='#10B981',
        prerequisites=[],
        steps=[
            # Step 1: Predict what happens with XOR and no hidden layer
            PathStep(
                step_number=1,
                problem_id='xor',
                title='Prediction: Can XOR Learn Without Hidden Layers?',
                learning_objectives=[
                    'Think critically about network architecture',
                    'Understand linear separability before seeing it',
                    'Build intuition through prediction'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='xor_no_hidden',
                    show_result_after=True
                ),
                hints=[
                    'Think about what a single neuron can compute',
                    'Consider: can you draw ONE line to separate XOR outputs?'
                ]
            ),
            # Step 2: Build the right architecture for AND gate
            PathStep(
                step_number=2,
                problem_id='and_gate',
                title='Build Challenge: AND Gate Architecture',
                learning_objectives=[
                    'Understand that AND is linearly separable',
                    'Learn that simple problems need simple networks',
                    'Practice building network architectures'
                ],
                step_type='build_challenge',
                build_challenge=BuildChallengeData(
                    min_layers=0,
                    max_layers=2,
                    min_hidden_neurons=0,
                    max_hidden_neurons=8,
                    must_have_hidden=False,
                    success_message='Perfect! AND only needs a single neuron!'
                ),
                hints=[
                    'AND is one of the simplest problems',
                    'Think: do you NEED hidden layers for this?'
                ]
            ),
            # Step 3: Build architecture for XOR
            PathStep(
                step_number=3,
                problem_id='xor',
                title='Build Challenge: XOR Architecture',
                learning_objectives=[
                    'Apply the lesson from prediction quiz',
                    'Understand why XOR needs hidden layers',
                    'Build a working XOR network'
                ],
                step_type='build_challenge',
                build_challenge=BuildChallengeData(
                    min_layers=1,
                    max_layers=3,
                    min_hidden_neurons=2,
                    max_hidden_neurons=8,
                    must_have_hidden=True,
                    success_message='Excellent! XOR requires hidden layers to work!'
                ),
                hints=[
                    'Remember: XOR is NOT linearly separable',
                    'You need at least one hidden layer',
                    'Try 2-4 neurons in the hidden layer'
                ]
            ),
            # Step 4: Debug challenge - find the zero init bug
            PathStep(
                step_number=4,
                problem_id='xor',
                title='Debug Challenge: Why Won\'t It Learn?',
                learning_objectives=[
                    'Diagnose common neural network bugs',
                    'Understand the symmetry problem',
                    'Learn about weight initialization'
                ],
                step_type='debug_challenge',
                debug_challenge=DebugChallengeData(
                    challenge_id='zero_init_bug'
                ),
                hints=[
                    'Look at the weight initialization setting',
                    'What happens when all weights start the same?'
                ]
            ),
            # Step 5: Predict what happens with high learning rate
            PathStep(
                step_number=5,
                problem_id='xor',
                title='Prediction: Learning Rate = 10?',
                learning_objectives=[
                    'Understand learning rate effects',
                    'Predict gradient descent behavior',
                    'Learn about stability vs speed tradeoff'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='high_lr',
                    show_result_after=True
                ),
                hints=[
                    'Learning rate controls step size in gradient descent',
                    'What happens if steps are TOO big?'
                ]
            ),
            # Step 6: Debug challenge - find the exploding loss bug
            PathStep(
                step_number=6,
                problem_id='xor',
                title='Debug Challenge: Loss Explodes!',
                learning_objectives=[
                    'Recognize learning rate problems',
                    'Diagnose numerical instability',
                    'Apply knowledge from prediction quiz'
                ],
                step_type='debug_challenge',
                debug_challenge=DebugChallengeData(
                    challenge_id='exploding_loss'
                ),
                hints=[
                    'The symptoms tell you a lot',
                    'This relates to what you just predicted!'
                ]
            ),
            # Step 7: Final training - put it all together
            PathStep(
                step_number=7,
                problem_id='xor',
                title='Victory Lap: Train XOR Successfully',
                learning_objectives=[
                    'Apply everything you learned',
                    'Successfully train XOR from scratch',
                    'Celebrate your understanding!'
                ],
                step_type='training',
                required_accuracy=0.95,
                hints=[
                    'Use what you learned: hidden layers, good init, reasonable LR',
                    'The default settings should work well now that you understand them!'
                ]
            )
        ]
    )
}


def get_all_paths() -> List[Dict]:
    """Get all learning paths as dictionaries"""
    return [path.to_dict() for path in LEARNING_PATHS.values()]


def get_path(path_id: str) -> Dict:
    """Get a specific learning path with step details"""
    if path_id not in LEARNING_PATHS:
        return None

    path = LEARNING_PATHS[path_id]
    result = path.to_dict()
    result['steps'] = path.get_step_details()  # Override count with detailed array
    return result
