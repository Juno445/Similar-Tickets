import os
from typing import List
import logging
from datetime import datetime

import streamlit as st

from src.config import FreshServiceConfig
from src.freshservice.freshservice_client import FreshServiceTicketAnalyzer
from src.similarity.ticket_similarity import Ticket

# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
REVIEWER = os.getenv("REVIEW_USER", "anonymous")
SIM_LOWER = float(os.getenv("REVIEW_SIM_LOWER", "0.65"))
SIM_UPPER = float(os.getenv("REVIEW_SIM_UPPER", "0.9"))
MAX_CANDIDATES = int(os.getenv("REVIEW_MAX", "100"))
DAYS_BACK = int(os.getenv("REVIEW_DAYS_BACK", "60"))

# Configure logger to print to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=True)
def _load_analyzer() -> FreshServiceTicketAnalyzer:
    """Singleton FreshServiceTicketAnalyzer (cached across reruns)."""
    cfg = FreshServiceConfig()
    return FreshServiceTicketAnalyzer(cfg)


def _load_candidates(analyzer: FreshServiceTicketAnalyzer):
    """Pull tickets & compute potential duplicates once per session."""
    with st.spinner("Fetching tickets from Freshservice …"):
        fs_tickets = analyzer.fetch_tickets_by_group(days_back=DAYS_BACK)
        tickets = analyzer.convert_freshservice_to_tickets(fs_tickets)
        logger.info(f"Fetched {len(tickets)} tickets from Freshservice.")
        st.toast(f"Fetched {len(tickets)} tickets.")

    if not tickets:
        logger.warning("No tickets fetched, so no candidates can be generated.")
        return []

    with st.spinner("Computing potential duplicates …"):
        # Add tickets to detector for analysis
        existing = {t.id for t in analyzer.detector.tickets}
        new_tickets = [t for t in tickets if t.id not in existing]
        if new_tickets:
            analyzer.detector.add_tickets_batch(new_tickets)
        
        # Use the same deduplicated approach as the webhook candidates endpoint
        # This avoids the A→B, B→A duplicate pairs issue
        trainer = analyzer.trainer
        pairs = trainer.get_unlabeled_pairs(min_similarity=SIM_LOWER, max_pairs=MAX_CANDIDATES * 2)
        logger.info(f"Found {len(pairs)} raw unlabeled pairs with similarity >= {SIM_LOWER:.2f}.")
        st.toast(f"Found {len(pairs)} raw unlabeled pairs.")
        
        # Get already processed pairs from session state
        processed_pairs = getattr(st.session_state, 'processed_pairs', set())
        
        # Convert pairs to the format expected by the UI
        candidates = []
        for t1, t2, similarity_score in pairs:
            if SIM_LOWER <= similarity_score < SIM_UPPER:
                # Skip if this pair has already been processed
                pair_key = (t1.id, t2.id)
                if pair_key in processed_pairs:
                    continue
                    
                # Skip if tickets are from different departments
                if (t1.department_id is not None and 
                    t2.department_id is not None and 
                    t1.department_id != t2.department_id):
                    continue
                
                # Calculate duplicate probability for this pair
                duplicate_probability = analyzer._calculate_duplicate_probability(t1, [(t2, similarity_score)])
                
                # Create a candidate object similar to TicketProbability
                candidate = type('Candidate', (), {
                    'ticket_id': t1.id,
                    'title': t1.title,
                    'description': t1.description[:200] + "..." if len(t1.description) > 200 else t1.description,
                    'duplicate_probability': duplicate_probability,
                    'most_similar_ticket_id': t2.id,
                    'most_similar_ticket_title': t2.title,
                    'most_similar_ticket_description': t2.description[:200] + "..." if len(t2.description) > 200 else t2.description,
                    'similarity_score': similarity_score,
                    'requester_email': t1.requester_email,
                    'department_id': t1.department_id,
                })()
                candidates.append(candidate)
        
        logger.info(f"Filtered down to {len(candidates)} candidates for review (similarity between {SIM_LOWER:.2f} and {SIM_UPPER:.2f}).")
        logger.info(f"Excluded {len(processed_pairs)} already processed pairs from this session.")
        st.toast(f"Found {len(candidates)} candidates for review.")

        # Sort by duplicate probability (highest first)
        candidates.sort(key=lambda c: c.duplicate_probability, reverse=True)

    final_candidates = candidates[:MAX_CANDIDATES]
    logger.info(f"Returning {len(final_candidates)} candidates from _load_candidates function")
    return final_candidates


# --------------------------------------------------------------------------- #
#  Main UI
# --------------------------------------------------------------------------- #

a = _load_analyzer()
candidates = _load_candidates(a)

# Add detailed logging about what we're about to display
logger.info(f"About to display UI with {len(candidates)} candidates")
if candidates:
    logger.info(f"First candidate: ticket_id={candidates[0].ticket_id}, prob={candidates[0].duplicate_probability:.2f}")
else:
    logger.warning("No candidates to display in UI")

st.title("🕵️ Duplicate Ticket Review Console")
st.markdown(
    f"Logged in as **{REVIEWER}** – showing tickets with probability between {SIM_LOWER:0.2f} and {SIM_UPPER:0.2f}."
)

# Add debug info to the UI
st.write(f"**Debug Info:** Found {len(candidates)} candidates for review")
st.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Add a refresh button to force reload
if st.button("🔄 Refresh Candidates"):
    st.cache_resource.clear()
    st.rerun()

# Add expanded debug information
with st.expander("🔍 Debug Information"):
    st.write(f"Configuration:")
    st.write(f"- REVIEWER: {REVIEWER}")
    st.write(f"- SIM_LOWER: {SIM_LOWER}")
    st.write(f"- SIM_UPPER: {SIM_UPPER}")
    st.write(f"- MAX_CANDIDATES: {MAX_CANDIDATES}")
    st.write(f"- DAYS_BACK: {DAYS_BACK}")
    
    # Session state info
    processed_pairs = getattr(st.session_state, 'processed_pairs', set())
    st.write(f"Session info:")
    st.write(f"- Processed pairs in this session: {len(processed_pairs)}")
    
    if candidates:
        st.write(f"Sample candidates (first 3):")
        for i, cand in enumerate(candidates[:3]):
            st.write(f"  {i+1}. Ticket #{cand.ticket_id} (prob: {cand.duplicate_probability:.3f}, sim: {cand.similarity_score:.3f})")
    
    # Add button to clear session state
    if st.button("🗑️ Clear Session (Reset Processed Pairs)"):
        if 'processed_pairs' in st.session_state:
            del st.session_state.processed_pairs
        st.success("Session cleared! Refresh to see all candidates again.")
        st.rerun()

if not candidates:
    st.success("No tickets require review – great job!")
    st.stop()

for cand in candidates:
    with st.expander(f"Ticket #{cand.ticket_id} – {cand.title}  (prob={cand.duplicate_probability:.2f})"):
        # Original ticket info
        t_url = f"https://{a.config.masked_domain}/a/tickets/{cand.ticket_id}"
        dup_url = (
            f"https://{a.config.masked_domain}/a/tickets/{cand.most_similar_ticket_id}"
            if cand.most_similar_ticket_id else None
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### Original Ticket [#{cand.ticket_id}]({t_url})")
            st.text(cand.title)
            st.write(cand.description)
        with col2:
            if cand.most_similar_ticket_id:
                st.markdown(f"### Potential Duplicate [#{cand.most_similar_ticket_id}]({dup_url})")
                st.text(cand.most_similar_ticket_title or "(no title)")
                st.write(cand.most_similar_ticket_description or "(no description)")
                st.write(f"Similarity score: **{cand.similarity_score:.3f}**")
            else:
                st.warning("No similar ticket found (should not happen).")

        # Action buttons - use both ticket IDs to ensure uniqueness
        approved = st.button("✅ Approve Merge", key=f"approve_{cand.ticket_id}_{cand.most_similar_ticket_id}")
        denied = st.button("❌ Not a Duplicate", key=f"deny_{cand.ticket_id}_{cand.most_similar_ticket_id}")

        if approved or denied:
            # Find existing Ticket objects with embeddings from the detector
            ticket_map = {t.id: t for t in a.detector.tickets}
            t1 = ticket_map.get(cand.ticket_id)
            t2 = ticket_map.get(cand.most_similar_ticket_id)
            
            if not t1 or not t2:
                st.error(f"Could not find tickets in detector: t1={cand.ticket_id}, t2={cand.most_similar_ticket_id}")
                continue
                
            is_dup = bool(approved)
            a.trainer.label_pair(
                t1,
                t2,
                is_duplicate=is_dup,
                confidence=1.0,
                labeled_by=REVIEWER,
            )

            if approved:
                merged = a.merge_ticket(cand.ticket_id, cand.most_similar_ticket_id)
                if merged:
                    st.success("Tickets merged successfully.")
                else:
                    st.error("Merge API call failed – label recorded anyway.")
            else:
                st.info("Marked as not duplicate.")
            
            # Remove this candidate from the session and refresh the page
            # This avoids re-fetching all tickets and recomputing everything
            if 'processed_pairs' not in st.session_state:
                st.session_state.processed_pairs = set()
            
            # Mark this pair as processed
            pair_key = (cand.ticket_id, cand.most_similar_ticket_id)
            st.session_state.processed_pairs.add(pair_key)
            st.session_state.processed_pairs.add((cand.most_similar_ticket_id, cand.ticket_id))  # Both directions
            
            st.success(f"✅ Processed pair: {cand.ticket_id} ↔ {cand.most_similar_ticket_id}")
            st.info("🔄 Refresh the page to see updated candidates list.")
            
            # Don't automatically reload - let user decide when to refresh 