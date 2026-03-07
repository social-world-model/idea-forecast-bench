import React from 'react';
import './About.css';

type Member = {
  name: string;
  role: string;
  photo: string; // e.g., /team-photos/haofei-yu.jpg
  initials: string; // fallback if image not found
  website: string;
};

const MEMBERS: Member[] = [
  {
    name: "Haofei Yu",
    role: "Core contributor",
    photo: "/team-photos/haofei.png",
    initials: "HY",
    website: "https://haofeiyu.me"
  },
  {
    name: "Fenghai Li",
    role: "Core contributor",
    photo: "/team-photos/fenghai.png",
    initials: "FH",
    website: "https://fenghaili.com"
  },
  {
    name: "Jiaxuan You",
    role: "Core Advisor",
    photo: "/team-photos/jiaxuan.png",
    initials: "JX",
    website: "https://cs.stanford.edu/~jiaxuan/"
  },
];

const About: React.FC = () => {
  return (
    <div className="about-container">
      <div className="about-header">
        <h1>About Us </h1>
        <p className="about-subtitle">
          Researchers at <a href="https://ulab-uiuc.github.io/" target="_blank" rel="noopener noreferrer" className="ulab-link">ULab</a> from the University of Illinois at Urbana-Champaign
        </p>
      </div>

      <section className="team-section">
        <ul className="team-grid" aria-label="Core Team Members">
          {MEMBERS.map((member) => (
            <li className="team-member" key={member.name}>
              <a
                href={member.website}
                className="member-avatar-link"
                target="_blank"
                rel="noopener noreferrer"
              >
                <div className="member-avatar">
                  <img
                    src={member.photo}
                    alt={member.name}
                    className="avatar-image"
                    onError={(e) => {
                      e.currentTarget.style.display = "none";
                      const fallback = e.currentTarget.nextElementSibling as HTMLElement | null;
                      if (fallback) fallback.style.display = "flex";
                    }}
                  />
                  <div className="avatar-fallback" style={{ display: "none" }} aria-hidden>
                    {member.initials}
                  </div>
                </div>
              </a>
              <a
                href={member.website}
                className="member-name-link"
                target="_blank"
                rel="noopener noreferrer"
              >
                <div className="member-name">{member.name}</div>
              </a>
              <div className="member-role">{member.role}</div>
            </li>
          ))}
        </ul>
      </section>

      <div className="project-section">
        <h2>About Our Project</h2>
        <div className="project-content">

          <div className="project-features">
            <h3><span className="highlight-text">Our Mission</span></h3>
            <p className="warning-paragraph">
              Live Idea Bench evaluates research-idea generation strategies on a shared paper stream so researchers can compare prompts, models, and heuristics under identical data windows and evaluation rules.
            </p>

            <h3><span className="highlight-text">Why historical backtesting?</span></h3>
            <p className="warning-paragraph">
              <span className="warning-text">Idea generation is easy to demo and hard to validate.</span> A one-off generation sample can look impressive even when it fails to anticipate what actually appears in later papers.
            </p>
            <p className="warning-paragraph">
              <span className="warning-text">Historical cutoff evaluation helps.</span> We replay the literature timeline, generate ideas from the papers available at each cutoff, and score whether those predictions match emerging future themes.
            </p>

            <h3><span className="highlight-text">Why daily evaluation?</span></h3>
            <p className="warning-paragraph">
              <span className="warning-text">Backtest alone is not enough.</span> Once a strategy has historical results, we keep generating fresh predictions and compare them against newly ingested papers to measure ongoing performance.
            </p>
            <p className="warning-paragraph">
              <span className="warning-text">This exposes drift.</span> Prompt quality, model quality, and topic dynamics all move over time; daily evaluation shows when a strategy stops generalizing.
            </p>

            <h3><span className="highlight-text">What is a strategy here?</span></h3>
            <p className="warning-paragraph">
              A strategy is an experiment configuration: generator type, model, prompt version, and generation parameters. It defines how ideas are produced from a fixed history of papers.
            </p>
            <p className="warning-paragraph">
              Some strategies are heuristic baselines, while others wrap an LLM. The benchmark compares them under the same cutoffs and evaluation logic.
            </p>

            <h3><span className="highlight-text">Disclaimer</span></h3>
            <p className="warning-paragraph">
              The content on this website is provided for general informational and <span className="warning-text">educational purposes only</span>. Generated ideas and benchmark scores are research artifacts, not statements of scientific truth or guarantees of future novelty.
            </p>
            <p className="warning-paragraph">
              Human review is still required before treating any generated idea as actionable research direction.
            </p>
          </div>

        </div>
      </div>

      <div className="contact-section">
        <h2>Join Us</h2>
        <p>
          Interested in contributing or have questions? We'd love to hear from you!
        </p>
        <div className="contact-buttons">
          <a href="mailto:jiaxuan@illinois.edu?cc=haofeiy2@illinois.edu" className="contact-button primary">
            Contact Us
          </a>
        </div>
      </div>
    </div>
  );
};

export default About;
