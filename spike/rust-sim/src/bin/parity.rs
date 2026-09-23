use laya_dino::game::Rng;
fn main() {
    let mut r = Rng::new(999);
    let v: Vec<u32> = (0..8).map(|_| r.next_u32()).collect();
    println!("rng {:?}", v);
    let mut g = laya_dino::game::Game::new(999);
    let mut kinds = vec![];
    for _ in 0..2000 { g.crashed = false; g.tick(laya_dino::game::Action::Run);
        if let Some(o) = g.obstacles.last() { let k=(o.kind.as_str(), o.w as i32);
            if kinds.last() != Some(&k) { kinds.push(k); } } }
    println!("layout {:?}", &kinds[..6.min(kinds.len())]);
}
