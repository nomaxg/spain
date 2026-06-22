use anyhow::{Context, Result, anyhow};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    io::{BufRead, BufReader, BufWriter, Stdin, Stdout, Write, stdin, stdout},
    time::{Duration, Instant},
};

pub trait Broker<M> {
    fn send_msg(&mut self, msg: &M) -> Result<(usize, Duration)>;
    fn receive_msg(&mut self) -> Result<(M, Duration)>;
}

pub struct JsonBroker {
    reader: BufReader<Stdin>,
    writer: BufWriter<Stdout>,
}

impl JsonBroker {
    pub fn new() -> Self {
        Self {
            reader: BufReader::new(stdin()),
            writer: BufWriter::new(stdout()),
        }
    }
}

impl Default for JsonBroker {
    fn default() -> Self {
        Self::new()
    }
}

impl<M> Broker<M> for JsonBroker
where
    M: Serialize + DeserializeOwned,
{
    fn send_msg(&mut self, msg: &M) -> Result<(usize, Duration)> {
        let timer = Instant::now();
        serde_json::to_writer(&mut self.writer, msg).context("failed to serialize message")?;
        self.writer
            .write_all(b"\n")
            .context("failed to write newline terminator")?;
        self.writer.flush().context("failed to flush writer")?;
        let time = timer.elapsed();
        let msg_size = serde_json::to_vec(msg)?.len();
        Ok((msg_size, time))
    }

    fn receive_msg(&mut self) -> Result<(M, Duration)> {
        let mut line = String::new();
        let bytes_read = self
            .reader
            .read_line(&mut line)
            .context("failed to read incoming message")?;

        if bytes_read == 0 {
            return Err(anyhow!("stdin closed while waiting for message"));
        }
        let trimmed = line.trim_end_matches(['\n', '\r']);
        let timer = Instant::now();
        let msg = serde_json::from_str(trimmed).context("failed to parse JSON message")?;
        Ok((msg, timer.elapsed()))
    }
}
