from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Any

from colorama import Back
from yaml import safe_load

# Set up logging
social_log = logging.getLogger(name="social")
social_log.setLevel("DEBUG")
now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
os.makedirs("./log", exist_ok=True)
file_handler = logging.FileHandler(f"./log/social-{str(now)}.log", encoding="utf-8")
file_handler.setLevel("DEBUG")
file_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
social_log.addHandler(file_handler)

# Add path to import from project root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from oasis.clock.clock import Clock
from oasis.social_agent.agents_generator import (
    gen_control_agents_with_data, 
    generate_reddit_agents
)
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

async def running(
    db_path: str,
    user_path: str,
    pair_path: str,
    round_post_num: int = 2,
    num_timesteps: int = 2,
    clock_factor: int = 10,
    recsys_type: str = "reddit",
    activate_prob: float = 0.2,
    show_score: bool = True,
    max_rec_post_len: int = 50,
    refresh_rec_post_count: int = 5,
    action_space_file_path: str = None,
    inference_configs: dict[str, Any] | None = None,
) -> None:
    """Run a simplified Reddit simulation.

    Args:
        db_path: Path to store the simulation database
        user_path: Path to JSON file containing user data
        pair_path: Path to JSON file containing initial posts
        round_post_num: Number of posts per timestep
        num_timesteps: Number of simulation timesteps
        clock_factor: Time acceleration factor
        recsys_type: Type of recommendation system
        activate_prob: Probability of agent activation per timestep
        show_score: Whether to show post scores
        max_rec_post_len: Maximum length of recommendation list
        refresh_rec_post_count: Number of posts to refresh
        action_space_file_path: Path to action space prompt file
        inference_configs: Configuration for the inference model
    """
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)

    # Initialize simulation components
    start_time = datetime(2024, 8, 6, 8, 0)
    clock = Clock(k=clock_factor)
    channel = Channel()
    
    # Load action space prompt
    with open(action_space_file_path, "r", encoding="utf-8") as file:
        action_space_prompt = file.read()

    # Initialize platform
    platform = Platform(
        db_path,
        channel,
        clock,
        start_time,
        allow_self_rating=False,  # Reddit doesn't allow self-rating
        show_score=show_score,
        recsys_type=recsys_type,
        max_rec_post_len=max_rec_post_len,
        refresh_rec_post_count=refresh_rec_post_count,
    )
    
    # Set up inference channel
    inference_channel = Channel()
    platform_task = asyncio.create_task(platform.running())

    # Generate agents
    agent_graph, id_mapping = await gen_control_agents_with_data(channel, 2)
    agent_graph = await generate_reddit_agents(
        user_path,
        channel,
        inference_channel,
        agent_graph,
        id_mapping,
        follow_post_agent=False,
        mute_post_agent=False,
        action_space_prompt=action_space_prompt,
        model_type=inference_configs["model_type"],
        is_openai_model=inference_configs.get("is_openai_model", True),
    )

    # Load initial posts
    with open(pair_path, "r") as f:
        posts = json.load(f)

    # Run simulation timesteps
    for timestep in range(num_timesteps):
        print(Back.GREEN + f"Timestep: {timestep + 1}" + Back.RESET)
        social_log.info(f"Timestep: {timestep + 1}")

        # Get posting and rating agents
        post_agent = agent_graph.get_agent(0)
        rate_agent = agent_graph.get_agent(1)

        # Create posts for this timestep
        async def create_post(i: int):
            post_index = i + timestep * round_post_num
            if post_index >= len(posts):
                return
            content = posts[post_index]["RC_1"]["body"]
            await post_agent.perform_action_by_data("create_post", content=content)

        # Create posts in parallel
        post_tasks = [create_post(i) for i in range(round_post_num)]
        await asyncio.gather(*post_tasks)

        # Update recommendation table
        await platform.update_rec_table()
        social_log.info("Updated recommendation table")

        # Let agents perform random actions
        action_tasks = []
        for _, agent in agent_graph.get_agents():
            if agent.user_info.is_controllable is False:
                if random.random() < activate_prob:
                    action_tasks.append(agent.perform_action_by_llm())
        
        random.shuffle(action_tasks)
        await asyncio.gather(*action_tasks)

    # Clean up
    await channel.write_to_receive_queue((None, None, ActionType.EXIT))
    await platform_task
    social_log.info("Simulation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simplified Reddit simulation.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="scripts/reddit_simple/config.yaml",
        help="Path to the YAML config file.",
    )
    
    args = parser.parse_args()
    
    with open(args.config_path, "r") as f:
        cfg = safe_load(f)
    
    asyncio.run(
        running(
            **cfg.get("data", {}),
            **cfg.get("simulation", {}),
            inference_configs=cfg.get("inference", {})
        )
    )