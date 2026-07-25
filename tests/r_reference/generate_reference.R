#!/usr/bin/env Rscript
#
# Generate ground-truth output from the original R implementation, for the
# Python port to be validated against.
#
#   Rscript tests/r_reference/generate_reference.R [output_dir]
#
# Sources the upstream R sources directly rather than requiring the package to
# be installed, so that no dependency on VennDiagram is needed.
#
# The R sources are NOT distributed with this Python package. Fetch them first:
#
#   git clone --depth 1 https://github.com/RRHO2/RRHO2 /tmp/RRHO2
#   cp -r /tmp/RRHO2/R .
#
# or set RRHO2_R_SOURCE to the directory holding RRHO2_initialize.R.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else "tests/r_reference/data"

find_r_dir <- function() {
  candidates <- c(Sys.getenv("RRHO2_R_SOURCE", unset = NA),
                  "R", "../R", "../../R")
  script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(script) == 1) {
    candidates <- c(candidates, file.path(dirname(script), "..", "..", "R"))
  }
  for (path in candidates) {
    if (!is.na(path) && file.exists(file.path(path, "RRHO2_initialize.R"))) {
      return(path)
    }
  }
  stop(paste0(
    "Could not find the upstream R sources.\n\n",
    "These are not distributed with the Python package. To fetch them:\n\n",
    "  git clone --depth 1 https://github.com/RRHO2/RRHO2 /tmp/RRHO2\n",
    "  cp -r /tmp/RRHO2/R .\n\n",
    "Then re-run this script from the repository root, or point\n",
    "RRHO2_R_SOURCE at the directory containing RRHO2_initialize.R.\n"
  ), call. = FALSE)
}
r_dir <- find_r_dir()
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
source(file.path(r_dir, "defaultStepSize.R"))
source(file.path(r_dir, "numericListOverlap.R"))
source(file.path(r_dir, "RRHO2_initialize.R"))

# ---------------------------------------------------------------------------
# Synthetic data, matching the scenario in the package README / vignette.
# ---------------------------------------------------------------------------
make_lists <- function(seed, nGenes, nDE, shuffle_second = FALSE) {
  set.seed(seed)
  Genes <- paste0("Genes", seq_len(nGenes))
  n_noise <- nGenes - 2 * nDE

  one_list <- function() {
    up <- runif(nDE, 0, 0.05)
    down <- runif(nDE, 0, 0.05)
    noise <- runif(n_noise, 0, 1)
    c(-log10(up),
      -log10(down) * (-1),
      -log10(noise) * sample(c(1, -1), n_noise, replace = TRUE))
  }

  l1 <- data.frame(Genes = Genes, DDE = one_list(), stringsAsFactors = FALSE)
  l2 <- data.frame(Genes = Genes, DDE = one_list(), stringsAsFactors = FALSE)
  if (shuffle_second) {
    l2 <- l2[sample(nrow(l2)), ]
  }
  list(l1 = l1, l2 = l2)
}

write_case <- function(name, l1, l2, ...) {
  obj <- RRHO2_initialize(l1, l2, ...)
  write.csv(l1, file.path(out_dir, paste0(name, "_list1.csv")), row.names = FALSE)
  write.csv(l2, file.path(out_dir, paste0(name, "_list2.csv")), row.names = FALSE)
  # write.table keeps 17 significant digits so the comparison is not limited by
  # the text round-trip.
  write.table(format(obj$hypermat, digits = 17),
              file.path(out_dir, paste0(name, "_hypermat.tsv")),
              sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
  for (q in c("uu", "dd", "ud", "du")) {
    gl <- obj[[paste0("genelist_", q)]]
    writeLines(gl[[1]], file.path(out_dir, paste0(name, "_", q, "_list1.txt")))
    writeLines(gl[[2]], file.path(out_dir, paste0(name, "_", q, "_list2.txt")))
    writeLines(gl[[3]], file.path(out_dir, paste0(name, "_", q, "_overlap.txt")))
  }
  cat(sprintf("%-28s dim=%dx%d  max=%.6f\n", name,
              nrow(obj$hypermat), ncol(obj$hypermat),
              max(obj$hypermat, na.rm = TRUE)))
  invisible(obj)
}

# Case 1: the documented example, defaults.
d <- make_lists(15213, 2000, 200)
write_case("base", d$l1, d$l2, labels = c("list1", "list2"))

# Case 2: same data, -log10 scale (the recommended setting).
write_case("log10", d$l1, d$l2, labels = c("list1", "list2"), log10.ind = TRUE)

# Case 3: coarser step size, wider separator strip.
write_case("step50_boundary25", d$l1, d$l2, stepsize = 50, boundary = 0.25)

# Case 4: Fisher / log odds ratio.
write_case("fisher", d$l1, d$l2, method = "fisher")

# Case 5: BH correction (small enough that R does not underflow to Inf).
small <- make_lists(7, 400, 40)
write_case("bh_small", small$l1, small$l2, multipleTesting = "BH")
write_case("by_small", small$l1, small$l2, multipleTesting = "BY")

# Case 6: strong enough that -log(p) exceeds 745, so exp(-x) underflows a
# float64. R's multipleTesting therefore returns Inf for those cells (and emits
# its own warning); the log-space correction in the Python port does not.
strong <- make_lists(1, 4000, 800)
write_case("strong", strong$l1, strong$l2)
write_case("strong_bh_legacy", strong$l1, strong$l2, multipleTesting = "BH")

# Case 7: gene lists given in different orders, and an odd list length.
odd <- make_lists(99, 777, 70, shuffle_second = TRUE)
write_case("shuffled_odd", odd$l1, odd$l2, log10.ind = TRUE)

# Case 8: tied scores, to pin down sort and peak tie-breaking.
set.seed(4242)
nGenes <- 600
Genes <- paste0("G", seq_len(nGenes))
tied1 <- round(rnorm(nGenes), 1)   # heavy ties
tied2 <- round(rnorm(nGenes), 1)
write_case("ties",
           data.frame(Genes = Genes, DDE = tied1, stringsAsFactors = FALSE),
           data.frame(Genes = Genes, DDE = tied2, stringsAsFactors = FALSE))

# ---------------------------------------------------------------------------
# Unit-level reference for numericListOverlap and defaultStepSize.
# ---------------------------------------------------------------------------
set.seed(31337)
n <- 300
s1 <- paste0("g", sample(n))
s2 <- paste0("g", sample(n))
for (m in c("hyper", "fisher")) {
  ov <- numericListOverlap(s1, s2, 17, method = m)
  write.table(format(ov$log.pval, digits = 17),
              file.path(out_dir, paste0("overlap_", m, "_logpval.tsv")),
              sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
  write.table(ov$counts, file.path(out_dir, paste0("overlap_", m, "_counts.tsv")),
              sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
}
writeLines(s1, file.path(out_dir, "overlap_sample1.txt"))
writeLines(s2, file.path(out_dir, "overlap_sample2.txt"))

step_grid <- expand.grid(n1 = c(1, 2, 10, 100, 101, 777, 2000, 20001),
                         n2 = c(1, 2, 10, 100, 101, 777, 2000, 20001))
step_grid$step <- mapply(function(a, b) {
  defaultStepSize(matrix(0, a, 1), matrix(0, b, 1))
}, step_grid$n1, step_grid$n2)
write.csv(step_grid, file.path(out_dir, "default_stepsize.csv"), row.names = FALSE)

cat("Reference written to", normalizePath(out_dir), "\n")
