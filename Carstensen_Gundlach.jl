
using DataFrames, CSV, InvertedIndices, Statistics, Random
using StatsPlots

##### Load and prepare dataset #####

df = CSV.read("Carstensen_Gundlach.csv", DataFrame, missingstring="-999.999")

# change column names to match paper
rename!(df, :kaufman => "rule", :mfalrisk => "malfal", :exprop2 => "exprop", :lngdpc95 => "lngdpc",
        :frarom => "trade", :lat => "latitude", :landsea => "coast")

# only keep required columns  
needed_columns = ["lngdpc", "rule", "malfal", "maleco", "lnmort", "frost", "humid",
                  "latitude", "eurfrac", "engfrac", "coast", "trade"]
df = df[:, needed_columns]

# drop all observations with missing values in the variables
dropmissing!(df)


# run analysis
include("functions.jl")

n = size(df, 1)
y = df.lngdpc .- mean(df.lngdpc)
X = [df.rule df.malfal]
Z = [ones(n) Matrix(df[:, needed_columns[Not(1:3)]])]

Σ = [1.0 1/2 1/2; 1/2 1.0 1/2; 1/2 1/2 1.0]


ll(y, X, Z, [-1.0, -1.0], ones(10, 2), Σ)



