//! 학습 과정 시각화
//!
//! 학습 진행 상황을 웹앱에서 표시할 수 있는 형태로 제공합니다.

use serde::{Serialize, Deserialize};
use std::time::{Duration, Instant};

/// 학습 통계
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingStats {
    /// 현재 에폭
    pub epoch: usize,
    /// 현재 스텝
    pub step: usize,
    /// 전체 스텝
    pub total_steps: usize,
    /// 손실 히스토리
    pub loss_history: Vec<f32>,
    /// 학습률 히스토리
    pub lr_history: Vec<f32>,
    /// 검증 손실 히스토리
    pub val_loss_history: Vec<f32>,
    /// Perplexity 히스토리
    pub ppl_history: Vec<f32>,
    /// 경과 시간 (초)
    pub elapsed_secs: f64,
    /// 추정 남은 시간 (초)
    pub remaining_secs: Option<f64>,
    /// 초당 토큰 처리량
    pub tokens_per_sec: f64,
}

/// 학습 시각화 유틸리티
pub struct TrainingViz {
    start_time: Instant,
    total_tokens: usize,
    stats: TrainingStats,
}

impl TrainingViz {
    /// 새로운 학습 시각화 생성
    pub fn new(total_steps: usize) -> Self {
        Self {
            start_time: Instant::now(),
            total_tokens: 0,
            stats: TrainingStats {
                epoch: 0,
                step: 0,
                total_steps,
                loss_history: Vec::new(),
                lr_history: Vec::new(),
                val_loss_history: Vec::new(),
                ppl_history: Vec::new(),
                elapsed_secs: 0.0,
                remaining_secs: None,
                tokens_per_sec: 0.0,
            },
        }
    }

    /// 스텝 업데이트
    pub fn update(&mut self, step: usize, loss: f32, lr: f32, batch_tokens: usize) {
        self.stats.step = step;
        self.stats.loss_history.push(loss);
        self.stats.lr_history.push(lr);
        self.stats.ppl_history.push(loss.exp());

        self.total_tokens += batch_tokens;
        self.stats.elapsed_secs = self.start_time.elapsed().as_secs_f64();

        if self.stats.elapsed_secs > 0.0 {
            self.stats.tokens_per_sec = self.total_tokens as f64 / self.stats.elapsed_secs;
        }

        // 남은 시간 추정
        if step > 0 {
            let secs_per_step = self.stats.elapsed_secs / step as f64;
            let remaining_steps = self.stats.total_steps.saturating_sub(step);
            self.stats.remaining_secs = Some(secs_per_step * remaining_steps as f64);
        }
    }

    /// 에폭 업데이트
    pub fn set_epoch(&mut self, epoch: usize) {
        self.stats.epoch = epoch;
    }

    /// 검증 손실 추가
    pub fn add_val_loss(&mut self, val_loss: f32) {
        self.stats.val_loss_history.push(val_loss);
    }

    /// 현재 통계 반환
    pub fn get_stats(&self) -> &TrainingStats {
        &self.stats
    }

    /// JSON으로 직렬화
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.stats).unwrap_or_default()
    }

    /// 학습 요약 문자열
    pub fn summary(&self) -> String {
        let stats = &self.stats;

        let remaining_str = match stats.remaining_secs {
            Some(secs) => format_duration(secs),
            None => "계산 중...".to_string(),
        };

        format!(
            r#"
═══════════════════════════════════════════
            학습 진행 상황
═══════════════════════════════════════════
  에폭:          {}/{}
  스텝:          {}/{}
  현재 손실:     {:.4}
  Perplexity:    {:.2}
  학습률:        {:.2e}
───────────────────────────────────────────
  경과 시간:     {}
  남은 시간:     {}
  처리 속도:     {:.0} tokens/sec
═══════════════════════════════════════════
"#,
            stats.epoch + 1, "?",
            stats.step, stats.total_steps,
            stats.loss_history.last().unwrap_or(&0.0),
            stats.ppl_history.last().unwrap_or(&1.0),
            stats.lr_history.last().unwrap_or(&0.0),
            format_duration(stats.elapsed_secs),
            remaining_str,
            stats.tokens_per_sec,
        )
    }
}

/// 시간을 읽기 쉬운 형태로 포맷
fn format_duration(secs: f64) -> String {
    let total_secs = secs as u64;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}

/// 학습 이벤트 (웹 스트리밍용)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TrainingEvent {
    /// 스텝 완료
    StepComplete {
        step: usize,
        loss: f32,
        lr: f32,
    },
    /// 에폭 완료
    EpochComplete {
        epoch: usize,
        avg_loss: f32,
        val_loss: Option<f32>,
    },
    /// 학습 완료
    TrainingComplete {
        total_steps: usize,
        final_loss: f32,
        duration_secs: f64,
    },
    /// 체크포인트 저장
    CheckpointSaved {
        path: String,
        step: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_viz() {
        let mut viz = TrainingViz::new(100);

        viz.update(1, 5.0, 0.001, 1000);
        viz.update(2, 4.5, 0.001, 1000);

        let stats = viz.get_stats();
        assert_eq!(stats.step, 2);
        assert_eq!(stats.loss_history.len(), 2);
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30.0), "30s");
        assert_eq!(format_duration(90.0), "1m 30s");
        assert_eq!(format_duration(3661.0), "1h 1m 1s");
    }
}
