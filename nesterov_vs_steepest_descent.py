import numpy as np
import time

class SteepestDescent:
    """Implementation of the Steepest Descent (Gradient Descent) optimization method."""
    
    def __init__(self, function, gradient):
        """
        Initialize Steepest Descent optimizer.
        
        Args:
            function: The objective function to minimize
            gradient: The gradient function of the objective
        """
        self.function = function
        self.gradient = gradient
        self.name = "Steepest Descent"
        
    def optimize(self, initial_point, learning_rate=0.01, max_iter=1000, tol=1e-6):
        """
        Run the Steepest Descent optimization algorithm.
        
        Args:
            initial_point: Starting point (numpy array)
            learning_rate: Step size for each iteration
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence (gradient norm)
            
        Returns:
            Dictionary containing result information
        """
        start_time = time.time()
        
        x = initial_point.copy()
        function_values = [self.function(x)]
        
        # Calculate initial gradient
        grad = self.gradient(x)
        grad_norm = np.linalg.norm(grad)
        
        iterations = 0
        gradient_evaluations = 1  # Count initial evaluation
        function_evaluations = 1  # Count initial evaluation
        
        # Main optimization loop
        for i in range(max_iter):
            iterations += 1
            
            # Update rule for steepest descent
            x = x - learning_rate * grad
            
            # Calculate new gradient
            grad = self.gradient(x)
            grad_norm = np.linalg.norm(grad)
            
            # Increment counters
            gradient_evaluations += 1
            function_evaluations += 1
            
            # Store function value
            current_value = self.function(x)
            function_values.append(current_value)
            
            # Check convergence
            if grad_norm < tol:
                break
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Prepare results
        results = {
            "final_point": x,
            "final_value": function_values[-1],
            "iterations": iterations,
            "gradient_evaluations": gradient_evaluations,
            "function_evaluations": function_evaluations,
            "elapsed_time": elapsed_time,
            "converged": grad_norm < tol
        }
        
        return results


class NesterovAcceleratedGradient:
    """Implementation of the Nesterov Accelerated Gradient optimization method."""
    
    def __init__(self, function, gradient):
        """
        Initialize Nesterov Accelerated Gradient optimizer.
        
        Args:
            function: The objective function to minimize
            gradient: The gradient function of the objective
        """
        self.function = function
        self.gradient = gradient
        self.name = "Nesterov Accelerated Gradient"
    
    def optimize(self, initial_point, learning_rate=0.01, momentum=0.9, max_iter=1000, tol=1e-6):
        """
        Run the Nesterov Accelerated Gradient optimization algorithm.
        
        Args:
            initial_point: Starting point (numpy array)
            learning_rate: Step size for each iteration
            momentum: Momentum coefficient
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence (gradient norm)
            
        Returns:
            Dictionary containing result information
        """
        start_time = time.time()
        
        x = initial_point.copy()
        v = np.zeros_like(x)  # Initialize velocity
        function_values = [self.function(x)]
        
        # Calculate initial gradient
        grad = self.gradient(x)
        grad_norm = np.linalg.norm(grad)
        
        iterations = 0
        gradient_evaluations = 1  # Count initial evaluation
        function_evaluations = 1  # Count initial evaluation
        
        # Main optimization loop
        for i in range(max_iter):
            iterations += 1
            
            # Nesterov look-ahead step: compute gradient at the "looked-ahead" position
            x_ahead = x + momentum * v
            grad = self.gradient(x_ahead)
            grad_norm = np.linalg.norm(grad)
            
            # Increment counters
            gradient_evaluations += 1
            
            # Update velocity and position
            v = momentum * v - learning_rate * grad
            x = x + v
            
            # Evaluate function at new position
            current_value = self.function(x)
            function_values.append(current_value)
            function_evaluations += 1
            
            # Check convergence
            if grad_norm < tol:
                break
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Prepare results
        results = {
            "final_point": x,
            "final_value": function_values[-1],
            "iterations": iterations,
            "gradient_evaluations": gradient_evaluations,
            "function_evaluations": function_evaluations,
            "elapsed_time": elapsed_time,
            "converged": grad_norm < tol
        }
        
        return results


def compare_optimizers(function_name, function, gradient, initial_point, params):
    """
    Compare the performance of Steepest Descent and Nesterov methods.
    
    Args:
        function_name: Name of the test function
        function: Objective function to minimize
        gradient: Gradient function
        initial_point: Starting point for optimization
        params: Dictionary of parameters for the optimizers
        
    Returns:
        None (prints comparison results)
    """
    print(f"\nComparing optimizers on {function_name}")
    print("-" * 60)
    
    # Initialize optimizers
    sd = SteepestDescent(function, gradient)
    nag = NesterovAcceleratedGradient(function, gradient)
    
    # Run Steepest Descent
    sd_results = sd.optimize(
        initial_point=initial_point,
        learning_rate=params.get("learning_rate", 0.01),
        max_iter=params.get("max_iter", 1000),
        tol=params.get("tol", 1e-6)
    )
    
    # Run Nesterov Accelerated Gradient
    nag_results = nag.optimize(
        initial_point=initial_point,
        learning_rate=params.get("learning_rate", 0.01),
        momentum=params.get("momentum", 0.9),
        max_iter=params.get("max_iter", 1000),
        tol=params.get("tol", 1e-6)
    )
    
    # Print comparison table
    print(f"{'Method':<25} {'Iterations':<12} {'Gradient Evals':<15} {'Time (s)':<10} {'Final Value':<15}")
    print("-" * 80)
    
    print(f"{sd.name:<25} {sd_results['iterations']:<12} {sd_results['gradient_evaluations']:<15} "
          f"{sd_results['elapsed_time']:.6f} {sd_results['final_value']:.8e}")
    
    print(f"{nag.name:<25} {nag_results['iterations']:<12} {nag_results['gradient_evaluations']:<15} "
          f"{nag_results['elapsed_time']:.6f} {nag_results['final_value']:.8e}")
    
    # Calculate speedup ratios
    iter_speedup = sd_results['iterations'] / nag_results['iterations']
    time_speedup = sd_results['elapsed_time'] / nag_results['elapsed_time']
    
    print("\nComputational Complexity Analysis:")
    print(f"- Steepest Descent: 1 gradient evaluation per iteration")
    print(f"- Nesterov: 1 gradient evaluation + momentum operations per iteration")
    print(f"- Iteration speedup: {iter_speedup:.2f}x")
    print(f"- Time speedup: {time_speedup:.2f}x")


# Define test functions
def quadratic_function(x):
    """Simple quadratic function f(x) = 0.5 * ||x||^2."""
    return 0.5 * np.sum(x**2)

def quadratic_gradient(x):
    """Gradient of the quadratic function."""
    return x

def rosenbrock_function(x):
    """Rosenbrock function f(x,y) = (1-x)^2 + 100(y-x^2)^2."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_gradient(x):
    """Gradient of the Rosenbrock function."""
    grad = np.zeros_like(x)
    grad[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    grad[1] = 200 * (x[1] - x[0]**2)
    return grad

def ill_conditioned_function(x):
    """Ill-conditioned quadratic function."""
    A = np.array([[100.0, 0.0], [0.0, 1.0]])
    return 0.5 * x.T @ A @ x

def ill_conditioned_gradient(x):
    """Gradient of the ill-conditioned function."""
    A = np.array([[100, 0], [0, 1]])
    return A @ x


def main():
    """Run optimization comparisons on test functions."""
    print("COMPARING STEEPEST DESCENT AND NESTEROV ACCELERATED GRADIENT")
    print("=" * 60)
    print("\nTheoretical Computational Complexity:")
    print("- Steepest Descent: O(1/ε) iterations, O(d) operations per iteration")
    print("- Nesterov: O(1/√ε) iterations, O(d) operations per iteration")
    print("  where ε is the desired accuracy and d is the problem dimension")
    
    # Test cases
    test_cases = [
        {
            "name": "Quadratic Function",
            "function": quadratic_function,
            "gradient": quadratic_gradient,
            "initial_point": np.array([10.0, 10.0]),
            "params": {"learning_rate": 0.01, "momentum": 0.9, "max_iter": 500, "tol": 1e-6}
        },
        {
            "name": "Rosenbrock Function",
            "function": rosenbrock_function,
            "gradient": rosenbrock_gradient,
            "initial_point": np.array([-1.2, 1.0]),
            "params": {"learning_rate": 0.001, "momentum": 0.9, "max_iter": 2000, "tol": 1e-6}
        },
        {
            "name": "Ill-Conditioned Function",
            "function": ill_conditioned_function,
            "gradient": ill_conditioned_gradient,
            "initial_point": np.array([10.0, 10.0]),
            "params": {"learning_rate": 0.01, "momentum": 0.95, "max_iter": 500, "tol": 1e-6}
        }
    ]
    
    # Run tests
    for case in test_cases:
        compare_optimizers(
            case["name"],
            case["function"],
            case["gradient"],
            case["initial_point"],
            case["params"]
        )

if __name__ == "__main__":
    main()
