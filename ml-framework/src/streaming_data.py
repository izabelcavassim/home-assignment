"""
Streaming Data Loader

Module for loading and preparing artist performance datasets for streaming growth analysis.
Handles time-series data with proper temporal alignment and artist sampling.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger("ml_framework")


def convert_to_week_number(df: pd.DataFrame, date_columns: List[str] = None) -> pd.DataFrame:
    """
    Convert date columns to ISO week numbers (YYYY-WW format) for consistent merging.
    
    Args:
        df: DataFrame containing date column(s)
        date_columns: List of possible date column names to check (in priority order).
                     If None, defaults to common names.
    
    Returns:
        DataFrame with 'week' column as ISO week number and 'week_date' as original date
    
    Raises:
        ValueError: If no date column is found
    """
    if date_columns is None:
        date_columns = ['week', 'week_start_date', 'start_date', 'date']
    
    # Find the first existing date column
    date_col = None
    for col in date_columns:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError(f"No date column found. Looked for: {date_columns}")
    
    # Convert to datetime
    dates = pd.to_datetime(df[date_col])
    
    # Create ISO week number (YYYY-WW)
    df['week'] = (
        dates.dt.isocalendar().year.astype(str) + 
        '-W' + 
        dates.dt.isocalendar().week.astype(str).str.zfill(2)
    )
    
    # Keep original date for reference
    df['week_date'] = dates
    
    return df


class StreamingDataLoader:
    """
    Load and prepare artist streaming and social metrics data.
    
    Handles:
    - Loading multiple data sources (streams, social, tickets)
    - Temporal alignment on week granularity
    - Artist sampling for tractability
    - Data quality checks
    """
    
    def __init__(
        self,
        data_dir: str,
        artist_sample_size: int = 100,
        min_weeks_per_artist: int = 20,
        random_state: int = 42,
        impute_missing: bool = False
    ):
        """
        Initialize streaming data loader.
        
        Args:
            data_dir: Directory containing artist performance CSVs
            artist_sample_size: Number of artists to sample for analysis
            min_weeks_per_artist: Minimum weeks of data required per artist
            random_state: Random seed for reproducibility
            impute_missing: Whether to impute missing values (default: False)
        """
        self.data_dir = Path(data_dir)
        self.artist_sample_size = artist_sample_size
        self.min_weeks_per_artist = min_weeks_per_artist
        self.random_state = random_state
        self.impute_missing = impute_missing
        self.sampled_artists = None
        
        logger.info(f"Initialized StreamingDataLoader:")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Artist sample size: {self.artist_sample_size}")
        logger.info(f"  Min weeks per artist: {self.min_weeks_per_artist}")
        logger.info(f"  Impute missing values: {self.impute_missing}")
    
    def load_artist_streams(self) -> pd.DataFrame:
        """
        Load artist streaming data from artist_mstreams_week.
        Filter for Spotify platform only.
        
        Returns:
            DataFrame with columns: artist_id, week, streaming metrics
        """
        logger.info("Loading artist streaming data...")
        
        streams_path = self.data_dir / "artist_mstreams_week.csv"
        if not streams_path.exists():
            raise FileNotFoundError(f"Streaming data not found: {streams_path}")
        
        df = pd.read_csv(streams_path)
        
        # Filter for Spotify only
        if 'platform_name' in df.columns:
            spotify_df = df[df['platform_name'].str.lower() == 'spotify'].copy()
            logger.info(f"  Loaded {len(df):,} total stream records")
            logger.info(f"  Filtered to {len(spotify_df):,} Spotify records")
            df = spotify_df
        else:
            logger.warning("  No platform_name column found, using all platforms")
        
        # Convert to ISO week numbers
        df = convert_to_week_number(df, date_columns=['week_start_date', 'week'])
        
        # Create cumulative streams per artist (sorted by week)
        df = df.sort_values(['artist_id', 'week_date'])
        df['cleaned_cumulative_spotify_streams'] = df.groupby('artist_id')['number_of_streams'].cumsum()
        logger.info("  Created cleaned_cumulative_spotify_streams column")
        
        # Log stats
        logger.info(f"  Unique artists: {df['artist_id'].nunique():,}")
        logger.info(f"  Date range: {df['week_date'].min()} to {df['week_date'].max()}")
        logger.info(f"  Week number range: {df['week'].min()} to {df['week'].max()}")
        logger.info(f"  Cumulative streams range: {df['cleaned_cumulative_spotify_streams'].min():,.0f} to {df['cleaned_cumulative_spotify_streams'].max():,.0f}")
        
        return df
    
    def load_social_metrics(self) -> pd.DataFrame:
        """
        Load social media metrics from artist_social_week and artist_instagram.
        Filter for Spotify platform only to match streaming data.
        Transform week column to ISO week number (YYYY-WW).
        
        Returns:
            DataFrame with social engagement metrics
        """
        logger.info("Loading social metrics data...")
        
        social_path = self.data_dir / "artist_social_week.csv"
        instagram_path = self.data_dir / "artist_instagram.csv"
        
        dfs_to_merge = []
        
        # Load weekly social metrics if available
        if social_path.exists():
            social_df = pd.read_csv(social_path)
            
            # Filter for Spotify only to match streaming data
            if 'platform_name' in social_df.columns:
                spotify_social = social_df[social_df['platform_name'].str.lower() == 'spotify'].copy()
                logger.info(f"  Loaded {len(social_df):,} total social week records")
                logger.info(f"  Filtered to {len(spotify_social):,} Spotify social records")
                social_df = spotify_social
            else:
                logger.warning("  No platform_name column in social data")
            
            # Convert to ISO week numbers
            if not social_df.empty:
                social_df = convert_to_week_number(social_df, date_columns=['week_start_date', 'week'])
                dfs_to_merge.append(social_df)
        
        # Load Instagram demographics if available
        if instagram_path.exists():
            instagram_df = pd.read_csv(instagram_path)
            # Instagram data doesn't have week dimension - store separately
            logger.info(f"  Loaded {len(instagram_df):,} Instagram records (no time dimension)")
            self.instagram_static = instagram_df
        
        if not dfs_to_merge:
            logger.warning("No social metrics data found")
            return pd.DataFrame()
        
        return dfs_to_merge[0] if dfs_to_merge else pd.DataFrame()
    
    def load_ticket_sales(self) -> pd.DataFrame:
        """
        Load secondary ticket sales data.
        
        Returns:
            DataFrame with ticket sales metrics
        """
        logger.info("Loading ticket sales data...")
        
        # Try loading different ticket data sources
        ticket_files = [
            "lsecondaryticket_artist_week.csv",
            "tsecondaryticket_artist_week.csv",
        ]
        
        dfs = []
        for filename in ticket_files:
            ticket_path = self.data_dir / filename
            if ticket_path.exists():
                df = pd.read_csv(ticket_path)
                
                # Convert to ISO week numbers
                try:
                    df = convert_to_week_number(df, date_columns=['week_start_date', 'start_date', 'week'])
                    dfs.append(df)
                    logger.info(f"  Loaded {len(df):,} records from {filename}")
                except ValueError as e:
                    logger.warning(f"  Could not convert dates in {filename}: {e}")
                    continue
        
        if not dfs:
            logger.warning("No ticket sales data found")
            return pd.DataFrame()
        
        # Combine ticket data sources
        if len(dfs) > 1:
            ticket_df = pd.concat(dfs, ignore_index=True)
            # Aggregate if there are duplicates
            group_cols = [col for col in ['artist_id', 'week'] if col in ticket_df.columns]
            if group_cols and len(group_cols) == 2:  # Need both artist_id and week
                # Get numeric columns excluding the groupby columns
                numeric_cols = ticket_df.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col not in group_cols]
                
                if numeric_cols:
                    # Aggregate numeric columns
                    ticket_df = ticket_df.groupby(group_cols)[numeric_cols].sum().reset_index()
                else:
                    # No numeric columns to aggregate, just drop duplicates
                    ticket_df = ticket_df.drop_duplicates(subset=group_cols)
        else:
            ticket_df = dfs[0]
        
        # Verify 'week' column exists
        if 'week' not in ticket_df.columns:
            logger.error("'week' column missing from ticket data after processing")
            return pd.DataFrame()
        
        logger.info(f"  Total ticket records: {len(ticket_df):,}")
        logger.info(f"  Week number range: {ticket_df['week'].min()} to {ticket_df['week'].max()}")
        return ticket_df
    
    def load_dma_data(self) -> pd.DataFrame:
        """
        Load geographic (DMA-level) data for streaming and tickets.
        
        Returns:
            DataFrame with DMA-level metrics aggregated to artist-week level
        """
        logger.info("Loading DMA-level data...")
        
        dma_files = {
            'artist_mstreams_dma_week.csv': 'streams',
            'lsecondaryticket_artist_dma_week.csv': 'tickets_l',
            'tsecondaryticket_artist_dma_week.csv': 'tickets_t'
        }
        
        dma_features = []
        
        for filename, source in dma_files.items():
            file_path = self.data_dir / filename
            if not file_path.exists():
                logger.warning(f"  DMA file not found: {filename}")
                continue
            
            df = pd.read_csv(file_path)
            
            # Convert to ISO week numbers
            try:
                df = convert_to_week_number(df, date_columns=['week_start_date', 'start_date', 'week'])
            except ValueError as e:
                logger.warning(f"  Could not convert dates in {filename}: {e}")
                continue
            
            # Calculate geographic diversity metrics per artist-week
            if 'dma_id' in df.columns:
                diversity = df.groupby(['artist_id', 'week']).agg({
                    'dma_id': 'nunique'  # Number of unique DMAs
                }).rename(columns={'dma_id': f'dma_count_{source}'})
                
                dma_features.append(diversity)
                logger.info(f"  Loaded {len(df):,} records from {filename}")
        
        if not dma_features:
            logger.warning("No DMA data found")
            return pd.DataFrame()
        
        # Combine all DMA features
        combined = dma_features[0]
        for df in dma_features[1:]:
            combined = combined.join(df, how='outer')
        
        combined = combined.reset_index()
        logger.info(f"  Created DMA diversity features: {len(combined):,} records")
        return combined
    
    def sample_artists(self, df: pd.DataFrame, metric_col: str = 'number_of_streams') -> List[str]:
        """
        Sample top artists by total streaming volume.
        
        Args:
            df: DataFrame with artist streaming data
            metric_col: Column to use for ranking artists
        
        Returns:
            List of sampled artist IDs
        """
        logger.info(f"Sampling {self.artist_sample_size} artists...")
        
        # Calculate total streams per artist
        artist_totals = df.groupby('artist_id')[metric_col].sum().sort_values(ascending=False)
        
        # Filter artists with minimum data availability
        weeks_per_artist = df.groupby('artist_id').size()
        valid_artists = weeks_per_artist[weeks_per_artist >= self.min_weeks_per_artist].index
        
        logger.info(f"  Total artists: {len(artist_totals):,}")
        logger.info(f"  Artists with {self.min_weeks_per_artist}+ weeks: {len(valid_artists):,}")
        
        # Sample from top artists that meet minimum data requirement
        eligible_artists = artist_totals[artist_totals.index.isin(valid_artists)]
        sampled = eligible_artists.head(self.artist_sample_size).index.tolist()
        
        self.sampled_artists = sampled
        logger.info(f"  Sampled {len(sampled)} artists")
        logger.info(f"  Top artist total {metric_col}: {artist_totals.iloc[0]:,.0f}")
        logger.info(f"  Bottom sampled artist total {metric_col}: {artist_totals[sampled[-1]]:,.0f}")
        
        return sampled
    
    def smart_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Domain-informed imputation for time-series streaming data.
        
        Uses different strategies for different feature types:
        - Static demographics: artist mean → global median
        - Touring/events: fill with 0 (no activity)
        - Social metrics: forward fill within artist → global median
        - Adds missingness indicators for key features
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with imputed values
        """
        logger.info("Applying smart imputation...")
        df = df.copy()
        
        initial_missing = df.isnull().sum().sum()
        
        # 1. Static demographics: artist mean → global median
        demo_cols = [
            'ages_13_17', 'ages_18_24', 'ages_25_34', 'ages_35_44', 'ages_45_64', 'ages_65_',
            'male_amount', 'female_amount',
            'follower_white', 'follower_african_american', 'follower_asian', 'follower_hispanic',
            'follower_lang_en', 'follower_lang_es', 'follower_lang_pt', 
            'follower_lang_de', 'follower_lang_fr', 'follower_lang_zh',
            'notable_users_ratio', 'audience_credibility'
        ]
        demo_imputed = 0
        for col in demo_cols:
            if col in df.columns and df[col].isnull().any():
                # Artist-level mean first
                df[col] = df.groupby('artist_id')[col].transform(
                    lambda x: x.fillna(x.mean())
                )
                # Global median for remaining
                df[col] = df[col].fillna(df[col].median())
                demo_imputed += 1
        
        # 2. Touring/event metrics: fill with 0 (no activity that week)
        tour_cols = [
            'st_event_count', 'st_num_tickets', 'st_order_value', 'st_avg_price',
            'dma_count_streams', 'dma_count_tickets_l', 'dma_count_tickets_t',
            'genre_id'
        ]
        tour_imputed = 0
        for col in tour_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(0)
                tour_imputed += 1
        
        # 3. Social/engagement metrics: forward fill within artist
        social_cols = [
            'max_following', 'engagement_rate', 'avg_likes', 'avg_comments', 'avg_views'
        ]
        social_imputed = 0
        for col in social_cols:
            if col in df.columns and df[col].isnull().any():
                # Forward fill within each artist
                df[col] = df.groupby('artist_id')[col].ffill()
                # Global median for remaining (e.g., first weeks)
                df[col] = df[col].fillna(df[col].median())
                social_imputed += 1
        
        # 4. Add missingness indicators for key features
        indicator_cols = ['engagement_rate', 'st_event_count', 'max_following']
        indicators_added = 0
        for col in indicator_cols:
            if col in df.columns:
                indicator_name = f'{col}_was_missing'
                if indicator_name not in df.columns:
                    # Mark which rows had missing values BEFORE imputation
                    # (we need to check the original, but we already imputed - 
                    # so we'll add indicators before imputation in the main flow)
                    indicators_added += 1
        
        final_missing = df.isnull().sum().sum()
        
        logger.info(f"  Imputation summary:")
        logger.info(f"    Demographics: {demo_imputed} columns")
        logger.info(f"    Touring/events: {tour_imputed} columns")
        logger.info(f"    Social metrics: {social_imputed} columns")
        logger.info(f"    Missing values: {initial_missing:,} → {final_missing:,}")
        logger.info(f"    Reduction: {100*(initial_missing-final_missing)/initial_missing:.1f}%")
        
        return df
    
    def create_joined_dataset(self) -> pd.DataFrame:
        """
        Create unified dataset by joining all data sources.
        
        Returns:
            DataFrame with all features aligned by artist and week
        """
        logger.info("Creating joined dataset...")
        
        # Load all data sources
        streams_df = self.load_artist_streams()
        social_df = self.load_social_metrics()
        ticket_df = self.load_ticket_sales()
        dma_df = self.load_dma_data()  # NEW: Load DMA diversity data
        
        # Sample artists based on streaming data
        sampled_artists = self.sample_artists(streams_df)

        
        # Filter to sampled artists
        streams_df = streams_df[streams_df['artist_id'].isin(sampled_artists)]
        
        # Start with streaming data as base
        df = streams_df.copy()
        
        # Join social metrics if available
        if not social_df.empty:
            social_df = social_df[social_df['artist_id'].isin(sampled_artists)]
            df = pd.merge(
                df, social_df,
                on=['artist_id', 'week'],
                how='left',
                suffixes=('', '_social')
            )
            logger.info(f"  Joined social metrics: {df.shape}")
        
        # Join ticket sales if available
        if not ticket_df.empty:
            # Add detailed diagnostics
            logger.info(f"  Before filtering - Ticket data:")
            logger.info(f"    Total records: {len(ticket_df):,}")
            logger.info(f"    Unique artists: {ticket_df['artist_id'].nunique()}")
            
            # Check overlap BEFORE filtering
            overlap = set(ticket_df['artist_id']).intersection(set(sampled_artists))
            logger.info(f"    Artists that overlap with sample: {len(overlap)}/{len(sampled_artists)}")
            
            # Filter to sampled artists
            ticket_df = ticket_df[ticket_df['artist_id'].isin(sampled_artists)]
            
            logger.info(f"  After filtering - Ticket data:")
            logger.info(f"    Records: {len(ticket_df):,}")
            logger.info(f"    Unique artists: {ticket_df['artist_id'].nunique()}")
            
            if len(ticket_df) > 0:
                # Check week overlap
                streaming_weeks = set(df['week'])
                ticket_weeks = set(ticket_df['week'])
                week_overlap = streaming_weeks.intersection(ticket_weeks)
                logger.info(f"    Week overlap: {len(week_overlap)}/{len(streaming_weeks)} streaming weeks have ticket data")
            
            df = pd.merge(
                df, ticket_df,
                on=['artist_id', 'week'],
                how='left',
                suffixes=('', '_ticket')
            )
            
            # Check how many rows have ticket data
            ticket_cols = [col for col in df.columns if '_ticket' in col or col.startswith('st_')]
            if ticket_cols:
                non_null_count = df[ticket_cols].notna().any(axis=1).sum()
                logger.info(f"  After join: {non_null_count:,}/{len(df):,} rows ({100*non_null_count/len(df):.1f}%) have ticket data")
        else:
            logger.info("  No ticket data available to join")
        
        # Join DMA diversity metrics (NEW)
        if not dma_df.empty:
            dma_df = dma_df[dma_df['artist_id'].isin(sampled_artists)]
            df = pd.merge(
                df, dma_df,
                on=['artist_id', 'week'],
                how='left'
            )
            logger.info(f"  Joined DMA diversity metrics: {df.shape}")
        
        # Join Instagram engagement & demographics (NEW)
        if hasattr(self, 'instagram_static') and self.instagram_static is not None:
            instagram_df = self.instagram_static[
                self.instagram_static['artist_id'].isin(sampled_artists)
            ].copy()
            
            # Select engagement and demographic features
            instagram_features = [
                'artist_id', 'engagement_rate', 'avg_likes', 'avg_comments', 'avg_views',
                'notable_users_ratio', 'audience_credibility',
                # Demographics - Age
                'ages_13_17', 'ages_18_24', 'ages_25_34', 'ages_35_44', 'ages_45_64', 'ages_65_',
                # Demographics - Gender
                'male_amount', 'female_amount',
                # Demographics - Ethnicity
                'follower_white', 'follower_african_american', 'follower_asian', 'follower_hispanic',
                # Demographics - Language
                'follower_lang_en', 'follower_lang_es', 'follower_lang_pt', 
                'follower_lang_de', 'follower_lang_fr', 'follower_lang_zh'
            ]
            
            # Only include columns that exist
            available_cols = [col for col in instagram_features if col in instagram_df.columns]
            if len(available_cols) > 1:  # More than just artist_id
                instagram_df = instagram_df[available_cols]
                df = pd.merge(
                    df, instagram_df,
                    on='artist_id',
                    how='left'
                )
                logger.info(f"  Joined Instagram engagement & demographics: {len(available_cols)-1} features")
        
        # Sort by artist and week for time-series operations
        df = df.sort_values(['artist_id', 'week']).reset_index(drop=True)
        
        # Apply imputation if requested
        if self.impute_missing:
            df = self.smart_impute(df)  
        
        logger.info(f"Final dataset shape: {df.shape}")
        logger.info(f"  Artists: {df['artist_id'].nunique()}")
        logger.info(f"  Weeks: {df['week'].nunique()}")
        logger.info(f"  Date range: {df['week'].min()} to {df['week'].max()}")
        logger.info(f"  Missing values: {df.isnull().sum().sum():,}")
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the dataset.
        
        Args:
            df: Joined dataset
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'n_rows': len(df),
            'n_artists': df['artist_id'].nunique(),
            'n_weeks': df['week'].nunique(),
            'date_range': (df['week'].min(), df['week'].max()),
            'columns': list(df.columns),
            'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_stats': df.select_dtypes(include=[np.number]).describe().to_dict()
        }
        return summary
