"""
Learning Paths for Neural Network Academy

Defines structured curricula guiding learners through the 32 problems
with clear progression, prerequisites, and educational scaffolding.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict

@dataclass
class PathStep:
    step_number: int
    problem_id: str
    title: str
    learning_objectives: List[str]
    required_accuracy: float = 0.95
    hints: List[str] = None

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
        return [
            {
                'stepNumber': step.step_number,
                'problemId': step.problem_id,
                'title': step.title,
                'learningObjectives': step.learning_objectives,
                'requiredAccuracy': step.required_accuracy,
                'hints': step.hints
            }
            for step in self.steps
        ]


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
        estimated_time='3-4 hours',
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
                problem_id='sine_wave',
                title='Sine Wave - Regression Basics',
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
                step_number=6,
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
                step_number=7,
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
                step_number=8,
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
