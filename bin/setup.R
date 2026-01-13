if (!require("renv", quietly = TRUE)) {
    install.packages("renv")
}

if (!file.exists("renv.lock")) {
    renv::init(bare = TRUE, restart = FALSE)
} else {
    renv::activate()
}

dependencies <- c(
    "mlogit",
    "tidyverse",
    "jsonlite",
    "lme4"
)

cat("Installing dependencies...\n")
renv::install(dependencies)

cat("Snapshotting environment...\n")
renv::snapshot(prompt = FALSE)

cat("R environment setup complete!\n")
