"""DNA-Protein Affinity Data Processor

Loads and processes DNA sequences, DNA-binding proteins (DBPs), and their
binding affinity data. Handles normalization and one-hot encoding.
"""

import numpy as np
import pandas
from pathlib import Path
from typing import Optional
import warnings


class DNAAffinityDataProcessor:
    """Process DNA sequences, DBPs, and building affinity data."""

    def __init__(
        self,
        sequences_path: str | Path = "../Data/training_seqs.txt",
        dbps_path: str | Path = "../Data/training_DBPs.txt",
        affinity_path: str | Path = "../Data/training_data.txt",
        sequence_length: int = 36,
        cleaned_data_dir: str | Path = "../cleaned_data",
    ):
        """Initialize paths and sequence configuration.

        Parameters
        ----------
        sequences_path : str | Path
            Path to file containing DNA sequences (one per line).
        dbps_path : str | Path
            Path to file containing DBP names (one per line).
        affinity_path : str | Path
            Path to file containing space-delimited affinity matrix.
        sequence_length : int
            Expected length of DNA sequences. Default: 36.
        cleaned_data_dir : str | Path
            Directory to store cleaned CSV data. Default: "../cleaned_data".
        """
        self.sequences_path = Path(sequences_path)
        self.dbps_path = Path(dbps_path)
        self.affinity_path = Path(affinity_path)
        self.sequence_length = sequence_length
        self.cleaned_data_dir = Path(cleaned_data_dir)

        self.sequences: list[str] = []
        self.dbps: list[str] = []
        self.df_affinity: Optional[pandas.DataFrame] = None
        self._base_to_index = {"A": 0, "C": 1, "G": 2, "T": 3}

    def _ensure_cleaned_data_dir(self) -> None:
        """Ensure cleaned data directory exists."""
        self.cleaned_data_dir.mkdir(parents=True, exist_ok=True)

    def _get_cleaned_data_path(self) -> Path:
        """Get path to cleaned data CSV file.

        Returns
        -------
        Path
            Path to cleaned data CSV file.
        """
        return self.cleaned_data_dir / "processed_affinity_data.csv"

    def _cleaned_data_exists(self) -> bool:
        """Check if cleaned data CSV already exists.

        Returns
        -------
        bool
            True if cleaned data CSV exists.
        """
        return self._get_cleaned_data_path().exists()

    def _load_cleaned_data(self) -> pandas.DataFrame:
        """Load cleaned data from CSV.

        Returns
        -------
        pandas.DataFrame
            Processed affinity dataframe.

        Raises
        ------
        FileNotFoundError
            If cleaned data file does not exist.
        """
        cleaned_path = self._get_cleaned_data_path()
        if not cleaned_path.exists():
            raise FileNotFoundError(f"Cleaned data not found at {cleaned_path}")
        print(f"Loading cleaned data from {cleaned_path}")
        return pandas.read_csv(cleaned_path)

    def _save_cleaned_data(self, df: pandas.DataFrame) -> None:
        """Save processed dataframe to CSV.

        Parameters
        ----------
        df : pandas.DataFrame
            Processed affinity dataframe to save.
        """
        self._ensure_cleaned_data_dir()
        cleaned_path = self._get_cleaned_data_path()
        print(f"Saving cleaned data to {cleaned_path}")
        df.to_csv(cleaned_path, index=False)

    def _validate_data_quality(self, df: pandas.DataFrame) -> None:
        """Perform smart data quality checks.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe to validate.

        Raises
        ------
        ValueError
            If data quality checks fail.
        """
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            raise ValueError(
                f"Found missing values:\n{missing_counts[missing_counts > 0]}"
            )

        # Check for duplicate rows (excluding DNA_onehot which is array-like)
        subset_cols = [c for c in df.columns if c != "DNA_onehot"]
        duplicates = df[subset_cols].duplicated().sum()
        if duplicates > 0:
            warnings.warn(f"Found {duplicates} duplicate rows in data")

        # Check affinity ranges
        if (df["Affinity"] == 0).all():
            warnings.warn("All affinity values are 0 - check input data")

        # Validate DNA sequences
        if "DNA probe" in df.columns:
            invalid_bases = df["DNA probe"].str.contains("[^ACGT]", regex=True).any()
            if invalid_bases:
                raise ValueError("Found invalid DNA bases (expected A, C, G, T only)")

        # Check one-hot encoding shape
        if "DNA_onehot" in df.columns:
            sample_encoding = df["DNA_onehot"].iloc[0]
            if isinstance(sample_encoding, np.ndarray):
                if sample_encoding.shape != (self.sequence_length, 4):
                    raise ValueError(
                        f"Invalid one-hot encoding shape: {sample_encoding.shape}, "
                        f"expected ({self.sequence_length}, 4)"
                    )

        print("✓ Data quality validation passed")


    def load_sequences(self) -> list[str]:
        """Load DNA sequences from file.

        Returns
        -------
        list[str]
            List of DNA sequences.
        """
        with self.sequences_path.open("r", encoding="utf-8") as handle:
            self.sequences = [line.strip() for line in handle if line.strip()]
        return self.sequences

    def load_dbps(self) -> list[str]:
        """Load DBP names from file.

        Returns
        -------
        list[str]
            List of DBP names.
        """
        with self.dbps_path.open("r", encoding="utf-8") as handle:
            self.dbps = [line.strip() for line in handle if line.strip()]
        return self.dbps

    def load_affinity_matrix(self) -> pandas.DataFrame:
        """Load raw affinity matrix from file.

        Returns
        -------
        pandas.DataFrame
            Raw affinity matrix (rows=sequences, columns=DBPs).

        Raises
        ------
        ValueError
            If affinity matrix shape doesn't match sequences and DBPs counts.
        """
        df_affinity_wide = pandas.read_csv(
            self.affinity_path,
            sep=r"\s+",
            engine="python",
            header=None,
        )

        if df_affinity_wide.shape != (len(self.sequences), len(self.dbps)):
            raise ValueError(
                "Affinity matrix shape mismatch: "
                f"{df_affinity_wide.shape} vs "
                f"({len(self.sequences)}, {len(self.dbps)})"
            )

        return df_affinity_wide

    def reshape_to_long_form(self, df_affinity_wide: pandas.DataFrame) -> pandas.DataFrame:
        """Reshape affinity matrix from wide to long format.

        Parameters
        ----------
        df_affinity_wide : pandas.DataFrame
            Wide-form affinity matrix.

        Returns
        -------
        pandas.DataFrame
            Long-form dataframe with columns: DNA probe, Protein, Affinity.
        """
        df_affinity = (
            df_affinity_wide
            .set_axis(self.dbps, axis=1)
            .assign(sequence=self.sequences)
            .melt(id_vars="sequence", var_name="Protein", value_name="Affinity")
            .rename(columns={"sequence": "DNA probe"})
        )
        return df_affinity

    def normalize_affinity_z_score(self, df_affinity: pandas.DataFrame) -> pandas.DataFrame:
        """Z-score normalize affinities per protein.

        Reduces outlier influence by normalizing within each protein group.

        Parameters
        ----------
        df_affinity : pandas.DataFrame
            Long-form affinity dataframe.

        Returns
        -------
        pandas.DataFrame
            Dataframe with added 'Affinity_z' column.
        """
        mean_by_protein = df_affinity.groupby("Protein")["Affinity"].transform("mean")
        std_by_protein = df_affinity.groupby("Protein")["Affinity"].transform("std")
        std_by_protein = std_by_protein.replace(0, np.nan)

        df_affinity["Affinity_z"] = (
            (df_affinity["Affinity"] - mean_by_protein) / std_by_protein
        )
        return df_affinity

    def validate_sequence_lengths(self, probes: pandas.Series) -> None:
        """Validate that all sequences have expected length.

        Parameters
        ----------
        probes : pandas.Series
            Series of DNA probe sequences.

        Raises
        ------
        ValueError
            If any sequence doesn't match expected length.
        """
        if not probes.str.len().eq(self.sequence_length).all():
            bad_lengths = probes.str.len().value_counts().to_dict()
            raise ValueError(
                f"Expected length {self.sequence_length} for all probes, "
                f"got lengths: {bad_lengths}"
            )

    def one_hot_encode_dna(self, seq: str) -> np.ndarray:
        """One-hot encode a DNA sequence.

        Parameters
        ----------
        seq : str
            DNA sequence (uppercase ACGT).

        Returns
        -------
        np.ndarray
            One-hot encoded array of shape (sequence_length, 4).

        Raises
        ------
        ValueError
            If sequence contains invalid bases.
        """
        encoded = np.zeros((self.sequence_length, 4), dtype=np.int8)
        for idx, base in enumerate(seq):
            base_index = self._base_to_index.get(base)
            if base_index is None:
                raise ValueError(f"Unexpected base '{base}' in sequence: {seq}")
            encoded[idx, base_index] = 1
        return encoded

    def encode_dna_sequences(self, probes: pandas.Series) -> np.ndarray:
        """One-hot encode all DNA sequences.

        Parameters
        ----------
        probes : pandas.Series
            Series of DNA probe sequences.

        Returns
        -------
        np.ndarray
            One-hot encoded sequences of shape (n_sequences, sequence_length, 4).
        """
        one_hot = np.stack(
            probes.map(self.one_hot_encode_dna).to_list(),
            axis=0
        )
        return one_hot

    def process(self) -> pandas.DataFrame:
        """Execute full data processing pipeline with intelligent caching.

        Checks if cleaned data CSV already exists. If it does, loads from cache.
        Otherwise, processes raw data, validates quality, saves to CSV, and returns.

        Returns
        -------
        pandas.DataFrame
            Processed affinity dataframe with columns:
            - DNA probe: DNA sequence
            - Protein: DBP name
            - Affinity: Raw binding affinity
            - Affinity_z: Z-score normalized affinity
            - DNA_onehot: One-hot encoded DNA sequence

        Raises
        ------
        ValueError
            If any validation step fails.
        """
        # Check if cleaned data already exists
        if self._cleaned_data_exists():
            self.df_affinity = self._load_cleaned_data()
            return self.df_affinity

        print("Processing raw data...")

        # Load raw data
        self.load_sequences()
        self.load_dbps()
        df_affinity_wide = self.load_affinity_matrix()

        # Reshape and normalize
        self.df_affinity = self.reshape_to_long_form(df_affinity_wide)
        self.df_affinity = self.normalize_affinity_z_score(self.df_affinity)

        # Encode DNA sequences
        probes = self.df_affinity["DNA probe"].astype(str).str.upper()
        self.validate_sequence_lengths(probes)
        one_hot = self.encode_dna_sequences(probes)

        # Add one-hot encoded sequences to dataframe
        self.df_affinity = self.df_affinity.assign(DNA_onehot=list(one_hot))

        # Validate data quality
        self._validate_data_quality(self.df_affinity)

        # Save cleaned data
        self._save_cleaned_data(self.df_affinity)

        return self.df_affinity

    def get_dataframe(self) -> pandas.DataFrame:
        """Get processed dataframe.

        Returns
        -------
        pandas.DataFrame
            Processed affinity dataframe. Calls process() if not already done.
        """
        if self.df_affinity is None:
            self.process()
        return self.df_affinity
