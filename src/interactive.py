import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import FloatSlider
from ipywidgets import interact


class InteractiveLinearRegression:
    intercept = 0
    coef = 0
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
        intercept_slider = FloatSlider(value=np.mean(y), min=-y.min() -1, max=y.max() + 1, step=0.01)
        coef_slider = FloatSlider(value=0, min=-1, max=1, step=0.0001)
        
        self.create_figure()
        
        interact(self.update_figure, intercept=intercept_slider, coef=coef_slider)
        
    def define_figure(self):
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set(title="Linear Regression", xlabel="X", ylabel="y")
        return self
        
    def create_figure(self):
        self.define_figure()
        self.scatter = self.ax.scatter(self.X, self.y, s=25, alpha=0.75, edgecolors="white")
        self.line, = self.ax.plot(self.X, self.f(), lw=2, color="red")
        mse = self.mse()
        self.formula = self.ax.text(x=0, y=26.5, s="$y = 0 + 0x_1$", fontsize=10)
        self.error = self.ax.text(x=0, y=25, s=f"MSE={mse:.2f}", fontsize=10)
        
    def update_figure(self, intercept, coef):
        self.intercept = intercept
        self.coef = coef
        self.line.set_ydata(self.f())
        mse = self.mse()
        self.formula.set_text(s=f"$y = {self.intercept:.2f} + {self.coef:.2f}x_1$")
        self.error.set_text(s=f"MSE={mse:.2f}")
        
        
    def f(self):
        return self.intercept + self.coef * self.X
    
    def mse(self):
        return np.sum((self.X - self.f())**2) / self.X.shape[0]
