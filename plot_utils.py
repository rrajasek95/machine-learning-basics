import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt, ceil

def plotAllDfColumns(dataFrame):
    assert type(dataFrame) == pd.DataFrame, "Not a pandas dataframe"
    
    numColumns = len(dataFrame.columns)
    gridSide = ceil(sqrt(numColumns))
    plotDimensions = (gridSide, gridSide)
    
    plt.figure(figsize=(10, 10))
    rowIndex = 0
    colIndex = 0
    for col in dataFrame.columns:
        plt.subplot2grid(plotDimensions, (rowIndex, colIndex))
        plt.title(col)
        dataFrame[col].hist()
        
        colIndex += 1
        if colIndex >= gridSide:
            colIndex = 0
            rowIndex += 1