#![allow(dead_code)]
pub struct Confidence {
    value: f64,
    age: u32,
    min: f64,
    max: f64,
    decay_rate: f64,
}

impl Confidence {
    pub fn new(initial: f64) -> Self {
        Self { value: initial.clamp(0.0, 1.0), age: 0, min: 0.0, max: 1.0, decay_rate: 0.01 }
    }
    pub fn with_decay(initial: f64, decay_rate: f64) -> Self {
        Self { value: initial.clamp(0.0, 1.0), age: 0, min: 0.0, max: 1.0, decay_rate }
    }
    pub fn value(&self) -> f64 { self.value }
    pub fn age(&self) -> u32 { self.age }
    pub fn update(&mut self, evidence: f64, weight: f64) {
        let w = weight.clamp(0.0, 1.0);
        self.value = self.value * (1.0 - w) + evidence.clamp(0.0, 1.0) * w;
        self.value = self.value.clamp(self.min, self.max);
    }
    pub fn decay(&mut self) {
        self.value = (self.value - self.decay_rate).max(self.min);
        self.age += 1;
    }
    pub fn fuse(a: f64, b: f64) -> f64 {
        if (a + b).abs() < 1e-10 { 0.0 } else { 2.0 * a * b / (a + b) }
    }
    pub fn fuse_bayesian(a: f64, b: f64) -> f64 {
        let inv = 1.0/a + 1.0/b;
        if inv.abs() < 1e-10 { 0.0 } else { 1.0 / inv }
    }
    pub fn clamp(v: f64, lo: f64, hi: f64) -> f64 { v.clamp(lo, hi) }
    pub fn is_expired(&self, max_age: u32) -> bool { self.age >= max_age }
    pub fn boost(&mut self, amount: f64) { self.value = (self.value + amount).min(self.max); }
    pub fn weaken(&mut self, amount: f64) { self.value = (self.value - amount).max(self.min); }
    pub fn reset(&mut self, initial: f64) { self.value = initial.clamp(self.min, self.max); self.age = 0; }
    pub fn score(&self) -> f64 { self.value * (1.0 - self.age as f64 * 0.01).max(0.0) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_new() { let c = Confidence::new(0.8); assert!((c.value() - 0.8).abs() < 1e-6); assert_eq!(c.age(), 0); }
    #[test] fn test_clamp_low() { let c = Confidence::new(-0.5); assert!(c.value() >= 0.0); }
    #[test] fn test_clamp_high() { let c = Confidence::new(1.5); assert!(c.value() <= 1.0); }
    #[test] fn test_update() { let mut c = Confidence::new(0.5); c.update(1.0, 0.5); assert!((c.value() - 0.75).abs() < 1e-6); }
    #[test] fn test_decay() { let mut c = Confidence::new(1.0); c.decay(); assert!(c.value() < 1.0); assert_eq!(c.age(), 1); }
    #[test] fn test_fuse() { assert!((Confidence::fuse(0.6, 0.4) - 0.48).abs() < 1e-6); }
    #[test] fn test_fuse_bayesian() { assert!((Confidence::fuse_bayesian(0.5, 0.5) - 0.25).abs() < 1e-6); }
    #[test] fn test_boost() { let mut c = Confidence::new(0.5); c.boost(0.3); assert!((c.value() - 0.8).abs() < 1e-6); }
    #[test] fn test_weaken() { let mut c = Confidence::new(0.5); c.weaken(0.3); assert!((c.value() - 0.2).abs() < 1e-6); }
    #[test] fn test_reset() { let mut c = Confidence::new(0.1); c.age = 50; c.reset(0.9); assert_eq!(c.age(), 0); assert!((c.value()-0.9).abs()<1e-6); }
    #[test] fn test_expired() { let mut c = Confidence::new(0.5); c.age = 100; assert!(c.is_expired(50)); assert!(!c.is_expired(200)); }
    #[test] fn test_score_fresh() { let c = Confidence::new(0.8); assert!(c.score() > 0.7); }
    #[test] fn test_score_aged() { let mut c = Confidence::new(0.8); for _ in 0..50 { c.decay(); } assert!(c.score() < c.value()); }
    #[test] fn test_with_decay() { let c = Confidence::with_decay(0.9, 0.05); assert!((c.value()-0.9).abs()<1e-6); }
    #[test] fn test_fuse_zero() { let r = Confidence::fuse(0.0, 0.5); assert!(r.abs() < 1e-6); }
    #[test] fn test_update_weight_clamp() { let mut c = Confidence::new(0.5); c.update(1.0, 2.0); assert!(c.value() <= 1.0); }
}