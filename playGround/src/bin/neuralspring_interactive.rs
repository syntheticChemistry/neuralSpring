// SPDX-License-Identifier: AGPL-3.0-or-later

//! Interactive experiment runner combining neuralSpring science with
//! Squirrel AI for conversational experiment analysis.
//!
//! Connects to both the neuralSpring primal (for science.* calls) and
//! Squirrel (for ai.query) to enable an interactive loop where users
//! run experiments and get AI-powered analysis of results.

#![expect(
    clippy::pedantic,
    clippy::nursery,
    reason = "playground binary — iterating rapidly"
)]

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::io::{AsyncBufReadExt, BufReader};

use neuralspring_playground::primal_client::PrimalClient;
use neuralspring_playground::squirrel_client::SquirrelClient;

#[derive(Parser)]
#[command(
    name = "neuralspring-interactive",
    about = "AI-driven interactive experiment runner for neuralSpring"
)]
struct Cli {
    /// Override neuralSpring primal socket path
    #[arg(long)]
    primal_socket: Option<PathBuf>,

    /// Override Squirrel socket path
    #[arg(long)]
    squirrel_socket: Option<PathBuf>,

    /// AI model to use (passed to Squirrel ai.query)
    #[arg(long, default_value = "default")]
    model: String,
}

struct Session {
    primal: PrimalClient,
    squirrel: Option<SquirrelClient>,
    model: String,
    last_result: Option<serde_json::Value>,
    last_capability: Option<String>,
    history: Vec<String>,
}

impl Session {
    async fn handle_command(&mut self, input: &str) -> Result<()> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Ok(());
        }

        self.history.push(trimmed.to_string());

        if trimmed == "help" || trimmed == "?" {
            print_help();
            return Ok(());
        }

        if trimmed == "capabilities" || trimmed == "caps" {
            return self.show_capabilities().await;
        }

        if trimmed == "health" {
            return self.show_health().await;
        }

        if trimmed == "providers" {
            return self.show_providers().await;
        }

        if let Some(rest) = trimmed.strip_prefix("run ") {
            return self.run_experiment(rest).await;
        }

        if trimmed == "analyze" {
            return self.analyze_last().await;
        }

        if let Some(rest) = trimmed.strip_prefix("ask ") {
            return self.ask_ai(rest).await;
        }

        // Default: treat as a natural language query to AI with context
        self.conversational_query(trimmed).await
    }

    async fn show_capabilities(&self) -> Result<()> {
        match self.primal.capability_list().await {
            Ok(caps) => {
                println!("neuralSpring capabilities ({}):", caps.len());
                for cap in &caps {
                    println!("  {cap}");
                }
            }
            Err(e) => eprintln!("Error listing capabilities: {e}"),
        }
        Ok(())
    }

    async fn show_health(&self) -> Result<()> {
        match self.primal.health().await {
            Ok(h) => println!("neuralSpring: {h}"),
            Err(e) => eprintln!("neuralSpring health error: {e}"),
        }
        if let Some(sq) = &self.squirrel {
            match sq.health().await {
                Ok(h) => println!("Squirrel: {} (uptime {}s)", h.status, h.uptime_secs),
                Err(e) => eprintln!("Squirrel health error: {e}"),
            }
        } else {
            println!("Squirrel: not connected");
        }
        Ok(())
    }

    async fn show_providers(&self) -> Result<()> {
        if let Some(sq) = &self.squirrel {
            match sq.list_providers().await {
                Ok(p) => println!("{}", serde_json::to_string_pretty(&p)?),
                Err(e) => eprintln!("Error: {e}"),
            }
        } else {
            eprintln!("No Squirrel connection — cannot list AI providers.");
        }
        Ok(())
    }

    async fn run_experiment(&mut self, spec: &str) -> Result<()> {
        let parts: Vec<&str> = spec.splitn(2, ' ').collect();
        let capability = parts[0];
        let params_str = parts.get(1).unwrap_or(&"{}");

        let full_cap = if capability.starts_with("science.") {
            capability.to_string()
        } else {
            format!("science.{capability}")
        };

        let params: serde_json::Value = match serde_json::from_str(params_str) {
            Ok(v) => v,
            Err(_) => {
                eprintln!("Invalid JSON params. Usage: run <capability> {{\"key\": value}}");
                return Ok(());
            }
        };

        eprint!("Running {full_cap}... ");
        match self.primal.call_capability(&full_cap, &params).await {
            Ok(result) => {
                println!("done.");
                println!("{}", serde_json::to_string_pretty(&result)?);
                self.last_result = Some(result);
                self.last_capability = Some(full_cap);
            }
            Err(e) => {
                println!("failed.");
                eprintln!("Error: {e}");
            }
        }
        Ok(())
    }

    async fn analyze_last(&self) -> Result<()> {
        let squirrel = match &self.squirrel {
            Some(s) => s,
            None => {
                eprintln!("No Squirrel connection — cannot analyze. Start Squirrel or use --squirrel-socket.");
                return Ok(());
            }
        };

        let (result, cap) = match (&self.last_result, &self.last_capability) {
            (Some(r), Some(c)) => (r, c),
            _ => {
                eprintln!("No experiment results to analyze. Run an experiment first.");
                return Ok(());
            }
        };

        let prompt = format!(
            "I ran the neuralSpring '{cap}' experiment and got these results:\n\n\
             ```json\n{}\n```\n\n\
             Please analyze these results. Explain what the key metrics mean, \
             whether the values are expected, and what they tell us about the \
             underlying physics or mathematics.",
            serde_json::to_string_pretty(result)?
        );

        eprint!("Sending to AI for analysis... ");
        match squirrel
            .ai_query(&prompt, Some(&self.model), Some(1024), Some(0.7))
            .await
        {
            Ok(resp) => {
                println!("done.\n");
                println!("{}", resp.response);
            }
            Err(e) => {
                println!("failed.");
                eprintln!("AI error: {e}");
            }
        }
        Ok(())
    }

    async fn ask_ai(&self, question: &str) -> Result<()> {
        let squirrel = match &self.squirrel {
            Some(s) => s,
            None => {
                eprintln!("No Squirrel connection.");
                return Ok(());
            }
        };

        eprint!("Thinking... ");
        match squirrel
            .ai_query(question, Some(&self.model), Some(1024), Some(0.7))
            .await
        {
            Ok(resp) => {
                println!("done.\n");
                println!("{}", resp.response);
            }
            Err(e) => {
                println!("failed.");
                eprintln!("AI error: {e}");
            }
        }
        Ok(())
    }

    async fn conversational_query(&self, input: &str) -> Result<()> {
        let squirrel = match &self.squirrel {
            Some(s) => s,
            None => {
                eprintln!("Unknown command: '{input}'. Type 'help' for commands.");
                eprintln!("(Connect Squirrel for natural language queries.)");
                return Ok(());
            }
        };

        let mut prompt = String::new();
        prompt.push_str(
            "You are a scientific computing assistant for neuralSpring, which validates \
             Anderson localization, spectral analysis, Hessian eigenanalysis, game theory, \
             and protein folding pipelines using the barraCuda GPU math engine.\n\n",
        );

        if let (Some(result), Some(cap)) = (&self.last_result, &self.last_capability) {
            prompt.push_str(&format!(
                "The user's most recent experiment was '{cap}' with results:\n```json\n{}\n```\n\n",
                serde_json::to_string_pretty(result).unwrap_or_default()
            ));
        }

        prompt.push_str(&format!("User question: {input}"));

        eprint!("Thinking... ");
        match squirrel
            .ai_query(&prompt, Some(&self.model), Some(1024), Some(0.7))
            .await
        {
            Ok(resp) => {
                println!("done.\n");
                println!("{}", resp.response);
            }
            Err(e) => {
                println!("failed.");
                eprintln!("AI error: {e}");
            }
        }
        Ok(())
    }
}

fn print_help() {
    println!(
        "\
Commands:
  run <capability> [params]  Run a science experiment via the primal
                             Example: run anderson_localization {{\"size\": 100, \"disorder\": 2.0}}
  analyze                    Send last experiment results to AI for analysis
  ask <question>             Ask the AI a direct question
  capabilities               List neuralSpring capabilities
  health                     Check primal and Squirrel health
  providers                  List available AI providers
  help                       Show this help
  quit / exit                Exit

Any other input is sent to AI as a conversational query with experiment context."
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let cli = Cli::parse();

    eprintln!("neuralSpring Interactive Experiment Runner");
    eprintln!("==========================================\n");

    let primal = match &cli.primal_socket {
        Some(path) => PrimalClient::new(path.clone()),
        None => PrimalClient::discover()
            .context("Could not find neuralSpring primal. Is neuralspring_primal running?")?,
    };

    eprintln!("[init] Connected to neuralSpring primal.");

    let squirrel = match &cli.squirrel_socket {
        Some(path) => Some(SquirrelClient::new(path.clone())),
        None => match SquirrelClient::discover() {
            Ok(s) => {
                eprintln!("[init] Connected to Squirrel (AI routing enabled).");
                Some(s)
            }
            Err(_) => {
                eprintln!("[init] Squirrel not found — AI features disabled.");
                eprintln!("[init] Start Squirrel or use --squirrel-socket to enable AI.\n");
                None
            }
        },
    };

    let mut session = Session {
        primal,
        squirrel,
        model: cli.model,
        last_result: None,
        last_capability: None,
        history: Vec::new(),
    };

    println!("Type 'help' for commands, 'quit' to exit.\n");

    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();

    loop {
        eprint!("> ");
        // Flush stderr prompt
        tokio::task::yield_now().await;

        match lines.next_line().await? {
            Some(ref line) => {
                let trimmed = line.trim().to_string();
                if trimmed == "quit" || trimmed == "exit" {
                    println!("Goodbye.");
                    break;
                }
                if let Err(e) = session.handle_command(&trimmed).await {
                    eprintln!("Error: {e}");
                }
                println!();
            }
            None => break,
        }
    }

    Ok(())
}
